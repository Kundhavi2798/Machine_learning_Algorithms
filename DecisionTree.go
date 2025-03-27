package main

import (
	"fmt"
	"os"
	"strings"
)

type DecisionNode struct {
	Question    string
	TrueBranch  *DecisionNode
	FalseBranch *DecisionNode
	Result      string
}

func Classify(node *DecisionNode, input string) string {
	if node.Result != "" {
		return node.Result
	}

	if strings.Contains(input, node.Question) {
		return Classify(node.TrueBranch, input)
	}
	return Classify(node.FalseBranch, input)
}

func main() {
	filePath := "/home/kundhavk/Machine_learning_Algorithms/shrug-pc11state-poly-shp/state.prj"

	data, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	content := string(data)

	tree := &DecisionNode{
		Question: "GCS_WGS_1984",
		TrueBranch: &DecisionNode{
			Result: "This is a WGS84 coordinate system.",
		},
		FalseBranch: &DecisionNode{
			Result: "This is NOT a WGS84 coordinate system.",
		},
	}

	result := Classify(tree, content)
	fmt.Println(result)
}
