# evaluate.py
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    return loss, accuracy