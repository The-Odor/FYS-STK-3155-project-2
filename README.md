### Outputs:
    Initial outputs, from the benchmarking of NeuralNetwork.py, SGD.py, or LogisticRegression.py, are found in parameterValues.
	Data that has been further analyzed, with dataAnalysis.py (in LaTeX/appendix) are found in LaTeX/appendix/analyzedParameters in text-format or LaTeX/images in image-format
	Initial outputs are manually duplicated into LaTeX/appendix for analysis with dataAnalysis.py


### Running the Code:
    NeuralNetwork.py: 	Requires activationFunctions.py, dataHandler.py, and miscFunctions.py in the active 
	                    folder to function. 
				        Will not write to file by default, but can, by editing the variables writeToFile along with
				        either parametrizeClassification or parametrizeRegression, test a set of parameters to find
				        optimal combination and write it to file

	SGD.py:			    Requires miscFunctions.py in the active folder to function.
				        Will not write to file by default, but can, by editing the variables writeToFile along with
				        either testSGD or testANA, test a set of parameters to find optimal combination and 
				        write it to file

	LogisticRegression.py:	Requires miscFunctions.py and activationFunctions.py in the active folder to function.
				            Will not write to file by default, but can, by editing the variables writeToFile and
				            testParams, test a set of parameters to find optimal combination and write it to file

	dataAnalysis.py		    Requires the output files from the prior 3 files in the active folder to function. 
	                        Will not write to file.
				            Is not user-friendly

	dataHandler.py, miscFunctions.py, and activationFunctions.py are not stand-alone files and do nothing on their own


### Structure
	- Main folder
		- README.md
		- Source files
		- parameterValues
			- Output from source files
		- LaTeX
			- LaTeX files for report
			- images
				- Images for analysis and report
			- appendix
				- dataAnalysis.py
				- Output from source files, manually copied over
				- analyzedParameters
					- Printout from dataAnalysis.py, manually copied