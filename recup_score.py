import os, sys

score_type = ["L1", "chi2"]
path_analyse = f"./results/analyse"




for score in score_type:

	print(f"\nConstruct score {score}")

	models = os.listdir(f"{path_analyse}/{score}")
	tests = list()

	for model in models:
		tests += os.listdir(f"{path_analyse}/{score}/{model}")

	tests = list(set(tests))

	print(tests)