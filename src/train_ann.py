import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt


def build_baseline_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


def build_deeper_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model


def build_cnn(input_dim):
    # Reshape input for CNN: (batch_size, input_dim, 1)
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


def plot_loss_curves(train_losses, val_losses, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def custom_training_loop(model, X_train, y_train, X_val,
                         y_val, epochs=100, batch_size=16, patience=15):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Cast labels to float32
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)).batch(batch_size)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        epoch_train_loss_avg = tf.keras.metrics.Mean()

        # Explicit Forward and Backward propagation for each batch
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                # Forward propagation
                predictions = model(x_batch, training=True)
                loss = loss_fn(tf.expand_dims(y_batch, -1), predictions)

            # Backward propagation
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            epoch_train_loss_avg.update_state(loss)

        # Validation
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        for x_batch_val, y_batch_val in val_dataset:
            val_predictions = model(x_batch_val, training=False)
            val_loss = loss_fn(
                tf.expand_dims(
                    y_batch_val, -1), val_predictions)
            epoch_val_loss_avg.update_state(val_loss)

        t_loss = epoch_train_loss_avg.result().numpy()
        v_loss = epoch_val_loss_avg.result().numpy()

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")

        # Early stopping logic
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_weights is not None:
        model.set_weights(best_weights)

    return train_losses, val_losses


def train_ann():
    models_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)),
        'models')
    charts_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)),
        'charts')
    os.makedirs(charts_dir, exist_ok=True)

    data_path = os.path.join(models_dir, 'data_splits.npz')
    if not os.path.exists(data_path):
        print(
            f"Data file not found at {data_path}. Please run data_prep.py first.")
        return

    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    input_dim = X_train.shape[1]

    print("\n--- Training Baseline MLP (Custom Loop) ---")
    baseline_model = build_baseline_mlp(input_dim)
    t_loss, v_loss = custom_training_loop(
        baseline_model, X_train, y_train, X_val, y_val, epochs=100)
    plot_loss_curves(
        t_loss,
        v_loss,
        "Baseline MLP Loss",
        os.path.join(
            charts_dir,
            'baseline_mlp_loss.png'))

    print("\n--- Training Deeper MLP (Custom Loop) ---")
    deeper_model = build_deeper_mlp(input_dim)
    t_loss_deep, v_loss_deep = custom_training_loop(
        deeper_model, X_train, y_train, X_val, y_val, epochs=100)
    plot_loss_curves(
        t_loss_deep,
        v_loss_deep,
        "Deeper MLP Loss",
        os.path.join(
            charts_dir,
            'deeper_mlp_loss.png'))

    print("\n--- Training CNN (Custom Loop) ---")
    # Reshape for CNN
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_val_cnn = np.expand_dims(X_val, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    cnn_model = build_cnn(input_dim)
    t_loss_cnn, v_loss_cnn = custom_training_loop(
        cnn_model, X_train_cnn, y_train, X_val_cnn, y_val, epochs=100)
    plot_loss_curves(
        t_loss_cnn,
        v_loss_cnn,
        "CNN Loss",
        os.path.join(
            charts_dir,
            'cnn_loss.png'))

    # Compare validation losses and save the best model
    val_losses = {
        'Baseline MLP': min(v_loss),
        'Deeper MLP': min(v_loss_deep),
        'CNN': min(v_loss_cnn)
    }

    best_model_name = min(val_losses, key=val_losses.get)
    print(f"\nBest model based on Validation Loss: {best_model_name}")

    model_save_path = os.path.join(models_dir, 'heart_disease_model.keras')
    if best_model_name == 'Baseline MLP':
        best_model = baseline_model
        best_model.save(model_save_path)
        X_test_eval = X_test
    elif best_model_name == 'Deeper MLP':
        best_model = deeper_model
        best_model.save(model_save_path)
        X_test_eval = X_test
    else:
        best_model = cnn_model
        best_model.save(model_save_path)
        X_test_eval = X_test_cnn

    preds_prob = best_model.predict(X_test_eval, verbose=0).flatten()
    preds_classes = (preds_prob > 0.5).astype(int)

    print(f"\n--- Best Model ({best_model_name}) Results on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, preds_classes):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, preds_prob):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds_classes):.4f}")

    print("\nModels trained and evaluated. Best model saved to models/heart_disease_model.keras.")


if __name__ == "__main__":
    train_ann()
