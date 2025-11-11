import pickle

with open("titanic_3feature_new.pkcls", "rb") as file:
    model = pickle.load(file)

print("Model type:", type(model))
print("Domain features:", [v.name for v in model.domain.attributes])
print("Target:", model.domain.class_var.name)
print(model.domain.attributes)