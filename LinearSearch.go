package main

import (
	"fmt"
	"log"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LinearRegression represents a simple linear regression model: Y = W*X + B
type LinearRegression struct {
	g      *gorgonia.ExprGraph
	w, b   *gorgonia.Node
	x, y   *gorgonia.Node
	pred   *gorgonia.Node
	vm     gorgonia.VM
	solver gorgonia.Solver
}

func NewLinearRegression() *LinearRegression {
	g := gorgonia.NewGraph()

	// Define weights and bias
	w := gorgonia.NewScalar(g, gorgonia.Float32, gorgonia.WithName("w"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	b := gorgonia.NewScalar(g, gorgonia.Float32, gorgonia.WithName("b"), gorgonia.WithInit(gorgonia.Zeroes()))

	// Define input and output
	x := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(5, 1), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(5, 1), gorgonia.WithName("y"))

	// Define prediction formula: Y_pred = W*X + B
	pred := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, w)), b))

	// Define Mean Squared Error loss
	loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, pred))))))

	// Compute gradients
	grads, err := gorgonia.Grad(loss, w, b)
	if err != nil {
		log.Fatal(errors.Wrap(err, "Gradient computation failed"))
	}
	fmt.Println("the grades", grads)

	// Create the VM (Tape Machine for execution)
	vm := gorgonia.NewTapeMachine(g)

	// Use a valid solver (Adam Optimizer)
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.01))

	return &LinearRegression{
		g:      g,
		w:      w,
		b:      b,
		x:      x,
		y:      y,
		pred:   pred,
		vm:     vm,
		solver: solver,
	}
}

func (lr *LinearRegression) Train(X, Y tensor.Tensor, epochs int) {
	for i := 0; i < epochs; i++ {
		// Load data
		gorgonia.Let(lr.x, X)
		gorgonia.Let(lr.y, Y)

		// Forward pass
		if err := lr.vm.RunAll(); err != nil {
			log.Fatal(errors.Wrap(err, "Failed to run VM"))
		}

		// Backpropagation using solver
		if err := lr.solver.Step(gorgonia.NodesToValueGrads([]*gorgonia.Node{lr.w, lr.b})); err != nil {
			log.Fatal(errors.Wrap(err, "Solver step failed"))
		}

		// Reset VM for next iteration
		lr.vm.Reset()
	}
}

func main() {
	// Sample data (5 samples)
	XData := tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4, 5}), tensor.WithShape(5, 1))
	YData := tensor.New(tensor.WithBacking([]float32{2, 4, 6, 8, 10}), tensor.WithShape(5, 1)) // Y = 2X (ideal linear relation)

	// Initialize and train model
	model := NewLinearRegression()
	model.Train(XData, YData, 1000) // Corrected call

	// Print trained parameters
	fmt.Println("Trained Weight:", model.w.Value())
	fmt.Println("Trained Bias:", model.b.Value())
}
