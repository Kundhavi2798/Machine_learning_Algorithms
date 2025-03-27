package main

import (
	"fmt"
	"os"
	"strings"
)

func LinearSearch(arr []string, target string) int {
	for i, value := range arr {
		if strings.Contains(value, target) {
			return i
		}
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

	lines := strings.Split(string(data), "\n")
	target := "WGS_1984"

	result := LinearSearch(lines, target)

	if result != -1 {
		fmt.Printf("Element '%s' found at line %d\n", target, result+1)
	} else {
		fmt.Printf("Element '%s' not found in the file\n", target)
	}
}
