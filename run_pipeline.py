from pipeline.ExampleGen import ExampleGen

train_path, test_path = ExampleGen(use_cache=False)()
print(train_path, test_path)