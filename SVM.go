package main

import (
	"fmt"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type SVMModel struct {
	Weights *mat.VecDense
	Bias    float64
}

func (svm *SVMModel) Predict(features *mat.VecDense) float64 {
	var result mat.VecDense
	result.MulElemVec(svm.Weights, features)
	score := mat.Sum(&result) + svm.Bias

	if score >= 0 {
		return 1
	}
	return -1
}

func main() {
	filePath := "/home/kundhavk/Machine_learning_Algorithms/shrug-pc11state-poly-shp/state.prj"

	data, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	content := string(data)

	features := mat.NewVecDense(1, nil)
	if strings.Contains(content, "GCS_WGS_1984") {
		features.SetVec(0, 1.0)
	} else {
		features.SetVec(0, -1.0)
	}

	svm := SVMModel{
		Weights: mat.NewVecDense(1, []float64{1.0}),
		Bias:    0.0,
	}

	prediction := svm.Predict(features)

	if prediction == 1 {
		fmt.Println("SVM Prediction: This is a WGS84 coordinate system.")
	} else {
		fmt.Println("SVM Prediction: This is NOT a WGS84 coordinate system.")
	}
}
