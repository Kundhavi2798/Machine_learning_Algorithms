package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
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

	// Create SVM Classifier (using kNN as an example)
	svm := knn.NewKnnClassifier("euclidean", "linear", 2)

	// Train model
	err = svm.Fit(trainData)
	if err != nil {
		fmt.Println("Error training model:", err)
		return
	}

	// Predict
	predictions, err := svm.Predict(testData)
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
