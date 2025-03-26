package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
)

func main() {
	// Load dataset (Use AIKOSHA dataset if available)
	data, err := base.ParseCSVToInstances("dataset.csv", true)
	if err != nil {
		fmt.Println("Error loading dataset:", err)
		os.Exit(1)
	}

	// Train-Test Split
	trainData, testData := base.InstancesTrainTestSplit(data, 0.7)

	// Create Decision Tree Classifier
	tree := trees.NewID3DecisionTree(0.6)

	// Train the model
	err = tree.Fit(trainData)
	if err != nil {
		fmt.Println("Error training model:", err)
		return
	}

	// Predict on test data
	predictions, err := tree.Predict(testData)
	if err != nil {
		fmt.Println("Error predicting:", err)
		return
	}

	// Evaluate the model
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		fmt.Println("Error evaluating model:", err)
		return
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
