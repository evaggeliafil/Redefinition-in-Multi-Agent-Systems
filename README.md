# Redefinition in Multi-Agent Systems

This repository contains all the necessary scripts and data to reproduce the experiments of the thesis. 
For each Multi-Agent Architecture, there is a corresponding folder, and within each of these folders there are two subfolders, one for each response format (FF and MC). Scripts for all four prompting techniques are included in the response-format folder. 

All the scripts correspond to constants redefinitions. To perform the units redefinitions:
- All the instances of the word "constant" should be changed to "unit".
- The constants dataset should be replaced with the corresponding units one:
  - redefinitionsCSV.csv to UnitsRedefinitions.csv
  - constantsrRED_MC.csv to UnitsRED_MC.csv
  - NR_constantRED_MC.csv to NR_MC_units.csv
All these .csv files are located in the Datasets folder.

The scripts use Qwen2.5-72B-Instruct as an example. To use a different model simply change the model id path. 

The Evaluation folder contains the evaluation script for each system and response format. 
