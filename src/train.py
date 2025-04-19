# train.py
def train_model(model, x_train, y_train, batch_size=64, epochs=10, val_split=0.2):
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split
    )
    return history