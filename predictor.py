import pickle
#Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model=load_model('pipeline.pkl')
#make predictions
input_features=[[2637,0.00073,0.127,17.22]]
def make_predictions(model, input_features):
    predict_class= model.predict(input_features)
    return predict_class

#print(make_predictions(model,input_features))