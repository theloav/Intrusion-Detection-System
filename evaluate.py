from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions_classes = predictions.argmax(axis=1)  # Get the predicted class
    report = classification_report(y_test, predictions_classes)
    print(report)
