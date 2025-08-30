import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import math
import datapy
import struct # Import the struct module

# --- Global Model Initialization ---
_model = None
def get_sentence_transformer_model():
    global _model
    if _model is None:
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")
    return _model

# --- Database Operations ---
def read_data_from_database():
    conn = sqlite3.connect('neural_network.db')
    cursor = conn.cursor()

    # Helper function to decode binary data if needed
    def decode_blob_to_float(blob_data):
        if isinstance(blob_data, bytes):
            # Assuming float32 (4 bytes) - adjust 'f' to 'd' for float64 (8 bytes) if needed
            return struct.unpack('<f', blob_data)[0]
        return float(blob_data) # Already a number or convertible string

    # Load hidden layer weights
    cursor.execute('''
    SELECT neuron_index, input_index, weight
    FROM hidden_layer_weights
    ORDER BY neuron_index, input_index
    ''')
    rows = cursor.fetchall()
    try:
        hidden_layer_weights = np.array([decode_blob_to_float(row[2]) for row in rows]).reshape(128, 384)
    except Exception as e: # Catch broader exception to see more details
        print(f"Error processing hidden layer weight: {e}")
        print(f"Problematic hidden layer weight rows sample: {rows[:5]}")
        raise

    # Load hidden layer biases
    cursor.execute('''
    SELECT bias FROM hidden_layer_biases
    ORDER BY neuron_index
    ''')
    rows = cursor.fetchall()
    try:
        hidden_layer_biases = np.array([decode_blob_to_float(row[0]) for row in rows])
    except Exception as e:
        print(f"Error processing hidden layer bias: {e}")
        print(f"Problematic hidden layer bias rows sample: {rows[:5]}")
        raise

    # Load output neuron weights
    cursor.execute('''
    SELECT weight FROM output_neuron_weights
    ORDER BY neuron_index
    ''')
    rows = cursor.fetchall()
    try:
        output_neuron_weights = np.array([decode_blob_to_float(row[0]) for row in rows])
    except Exception as e:
        print(f"Error processing output neuron weight: {e}")
        print(f"Problematic output neuron weight rows sample: {rows[:5]}")
        raise

    # Load output neuron bias
    cursor.execute('SELECT bias FROM output_neuron_biases LIMIT 1')
    row = cursor.fetchone()
    output_neuron_bias = decode_blob_to_float(row[0]) if row and row[0] is not None else 0.0

    conn.close()
    return hidden_layer_weights, hidden_layer_biases, output_neuron_weights, output_neuron_bias


def write_data_to_database(hidden_weights, hidden_biases, output_weights, output_bias):
    """
    Writes updated neural network weights and biases back to the 'neural_network.db' database.
    Encodes float data to BLOB for consistent storage if that's the existing pattern.
    Uses executemany for batch updates.
    """
    conn = sqlite3.connect('neural_network.db')
    cursor = conn.cursor()

    # Helper function to encode float to binary data
    def encode_float_to_blob(value):
        # Assuming float32 (4 bytes) - adjust 'f' to 'd' for float64 (8 bytes) if needed
        return struct.pack('<f', float(value))

    # Prepare data for executemany - ensure they are in list of tuples format
    hidden_weights_data = []
    if isinstance(hidden_weights, np.ndarray):
        for i in range(hidden_weights.shape[0]):
            for j in range(hidden_weights.shape[1]):
                hidden_weights_data.append((encode_float_to_blob(hidden_weights[i, j]), i, j))
    else:
        for i, neuron_weights in enumerate(hidden_weights):
            for j, weight in enumerate(neuron_weights):
                hidden_weights_data.append((encode_float_to_blob(weight), i, j))


    hidden_biases_data = [(encode_float_to_blob(bias), i) for i, bias in enumerate(hidden_biases)]
    output_weights_data = [(encode_float_to_blob(weight), i) for i, weight in enumerate(output_weights)]

    # Use executemany for faster updates
    cursor.executemany('''
        UPDATE hidden_layer_weights
        SET weight = ?
        WHERE neuron_index = ? AND input_index = ?
    ''', hidden_weights_data)

    cursor.executemany('''
        UPDATE hidden_layer_biases
        SET bias = ?
        WHERE neuron_index = ?
    ''', hidden_biases_data)

    cursor.executemany('''
        UPDATE output_neuron_weights
        SET weight = ?
        WHERE neuron_index = ?
    ''', output_weights_data)

    cursor.execute('''
        UPDATE output_neuron_biases
        SET bias = ?
        WHERE rowid = 1
    ''', (encode_float_to_blob(output_bias),))

    conn.commit()
    conn.close()

# --- Activation Functions ---
def sigmoid(x):
    """Applies the sigmoid activation function element-wise to a NumPy array."""
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Calculates the derivative of the sigmoid function element-wise."""
    s = sigmoid(x)
    return s * (1 - s)

# --- Embedding Generation ---
def get_vector(sentence: str):
    """
    Generates a single sentence embedding. Used for individual predictions
    or if batching is not desired for some reason (though batching is preferred).
    """
    model = get_sentence_transformer_model()
    return model.encode(sentence)

def get_vectors_batch(sentences: list):
    """
    Generates sentence embeddings for a list of sentences in a batch.
    This is highly recommended for performance.
    """
    model = get_sentence_transformer_model()
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
    return embeddings

# --- Loss Function ---
def cost_function(actual, predicted):
    """Calculates the squared error loss."""
    return (predicted - actual) ** 2

# --- Neural Network Core Logic ---
def forward_pass_batch(input_embeddings_batch: np.ndarray):
    """
    Performs a forward pass for a batch of input embeddings through the neural network.
    Returns all intermediate values needed for backpropagation.

    Args:
        input_embeddings_batch (np.ndarray): A 2D NumPy array of shape (batch_size, input_dim).

    Returns:
        tuple: (input_embeddings_batch, hidden_layer_outputs, hidden_layer_z_values, output_z, predicted_outputs_batch)
    """
    hidden_layer_weights, hidden_layer_biases, output_neuron_weights, output_neuron_bias = read_data_from_database()

    hidden_layer_z_values = np.dot(input_embeddings_batch, hidden_layer_weights.T) + hidden_layer_biases
    hidden_layer_outputs = sigmoid(hidden_layer_z_values)

    output_z = np.dot(hidden_layer_outputs, output_neuron_weights) + output_neuron_bias
    predicted_outputs_batch = sigmoid(output_z)

    return input_embeddings_batch, hidden_layer_outputs, hidden_layer_z_values, output_z, predicted_outputs_batch


def back_propagation_batch(input_embeddings_batch: np.ndarray, actual_labels_batch: np.ndarray,
                           predicted_outputs_batch: np.ndarray, hidden_layer_outputs: np.ndarray,
                           hidden_layer_z_values: np.ndarray, output_z: np.ndarray, learning_rate: float = 0.01):
    """
    Performs backpropagation to update weights and biases for a batch of data.

    Args:
        input_embeddings_batch (np.ndarray): Input embeddings for the batch (batch_size, input_dim).
        actual_labels_batch (np.ndarray): True labels for the batch (batch_size,).
        predicted_outputs_batch (np.ndarray): Predicted outputs from the forward pass (batch_size,).
        hidden_layer_outputs (np.ndarray): Outputs of the hidden layer before final activation (batch_size, num_hidden).
        hidden_layer_z_values (np.ndarray): Pre-activation values (z) of the hidden layer (batch_size, num_hidden).
        output_z (np.ndarray): Pre-activation values (z) of the output layer (batch_size,).
        learning_rate (float): The learning rate for gradient descent.
    """
    hidden_layer_weights, hidden_layer_biases, output_neuron_weights, output_neuron_bias = read_data_from_database()

    actual_labels_batch = actual_labels_batch.astype(np.float32)

    dC_dpredicted = 2 * (predicted_outputs_batch - actual_labels_batch)
    dpredicted_doutput_z = sigmoid_derivative(output_z)
    delta_output = dC_dpredicted * dpredicted_doutput_z
    dC_db_output = np.sum(delta_output)
    dC_dw_output = np.dot(hidden_layer_outputs.T, delta_output)

    dC_dhidden_out_per_sample = delta_output[:, np.newaxis] * output_neuron_weights[np.newaxis, :]
    dhidden_out_dhidden_z = sigmoid_derivative(hidden_layer_z_values)
    delta_hidden = dC_dhidden_out_per_sample * dhidden_out_dhidden_z
    dC_db_hidden = np.sum(delta_hidden, axis=0)
    dC_dw_hidden = np.dot(delta_hidden.T, input_embeddings_batch)

    output_neuron_bias -= learning_rate * dC_db_output
    output_neuron_weights -= learning_rate * dC_dw_output
    hidden_layer_biases -= learning_rate * dC_db_hidden
    hidden_layer_weights -= learning_rate * dC_dw_hidden

    write_data_to_database(hidden_layer_weights, hidden_layer_biases, output_neuron_weights, output_neuron_bias)


# --- Main Training Loop ---
if __name__ == "__main__":
    # --- Create/Initialize Database (Run this once if your DB is empty or corrupted) ---
    # You would typically run this in a separate script or once before training
    # to ensure the database has the correct schema and initial values.
    # If your DB already exists with the correct schema and values, skip this.
    # If you're consistently getting errors due to corrupt data, you might need
    # to delete 'neural_network.db' and run this to recreate it.

    conn = sqlite3.connect('neural_network.db')
    cursor = conn.cursor()

    # Drop tables if they exist to start fresh (ONLY FOR TESTING/INITIAL SETUP)
    cursor.execute("DROP TABLE IF EXISTS hidden_layer_weights")
    cursor.execute("DROP TABLE IF EXISTS hidden_layer_biases")
    cursor.execute("DROP TABLE IF EXISTS output_neuron_weights")
    cursor.execute("DROP TABLE IF EXISTS output_neuron_biases")

    # Create tables with appropriate types (REAL for floats, INTEGER for indices)
    # Using BLOB to store raw binary floats consistently if that's what's happening
    # but REAL is generally preferred if the data is purely numeric.
    # If you were getting this error because your previous data was BLOBs,
    # but you prefer to store floats as REAL, then change BLOB to REAL here
    # and remove struct.pack/unpack from read/write functions.
    # For now, let's assume BLOB to match your error.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS hidden_layer_weights (
        neuron_index INTEGER,
        input_index INTEGER,
        weight BLOB,
        PRIMARY KEY (neuron_index, input_index)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS hidden_layer_biases (
        neuron_index INTEGER PRIMARY KEY,
        bias BLOB
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS output_neuron_weights (
        neuron_index INTEGER PRIMARY KEY,
        weight BLOB
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS output_neuron_biases (
        bias BLOB
    )
    ''')
    conn.commit()

    # Initialize weights and biases if tables are empty
    # Get dimensions for the model: 128 hidden neurons, 384 input features (from 'all-MiniLM-L6-v2')
    num_hidden_neurons = 128
    input_dim = 384

    # Check if hidden_layer_weights is empty
    cursor.execute("SELECT COUNT(*) FROM hidden_layer_weights")
    if cursor.fetchone()[0] == 0:
        print("Initializing neural network weights and biases in the database...")
        # Initialize with small random values
        initial_hidden_weights = np.random.randn(num_hidden_neurons, input_dim) * 0.01
        initial_hidden_biases = np.random.randn(num_hidden_neurons) * 0.01
        initial_output_weights = np.random.randn(num_hidden_neurons) * 0.01
        initial_output_bias = np.random.randn(1)[0] * 0.01

        # Prepare data for insertion (using the same encoding as write_data_to_database)
        insert_hidden_weights_data = []
        for i in range(num_hidden_neurons):
            for j in range(input_dim):
                insert_hidden_weights_data.append((i, j, struct.pack('<f', initial_hidden_weights[i, j])))

        insert_hidden_biases_data = [(i, struct.pack('<f', initial_hidden_biases[i])) for i in range(num_hidden_neurons)]
        insert_output_weights_data = [(i, struct.pack('<f', initial_output_weights[i])) for i in range(num_hidden_neurons)]
        insert_output_bias_data = (struct.pack('<f', initial_output_bias),)

        cursor.executemany("INSERT INTO hidden_layer_weights (neuron_index, input_index, weight) VALUES (?, ?, ?)", insert_hidden_weights_data)
        cursor.executemany("INSERT INTO hidden_layer_biases (neuron_index, bias) VALUES (?, ?)", insert_hidden_biases_data)
        cursor.executemany("INSERT INTO output_neuron_weights (neuron_index, weight) VALUES (?, ?)", insert_output_weights_data)
        cursor.execute("INSERT INTO output_neuron_biases (bias) VALUES (?)", insert_output_bias_data)
        conn.commit()
        print("Database initialized with random weights.")
    else:
        print("Database already contains weights and biases. Skipping initialization.")
    conn.close()
    # --- End Database Initialization ---

    texts, labels = datapy.get_data()
    epochs = 5
    learning_rate = 0.01
    batch_size = 32

    all_labels = np.array(labels)

    get_sentence_transformer_model() # Load model once

    print("Starting neural network training...")

    for epoch in range(epochs):
        total_loss = 0
        num_samples = len(texts)

        permutation = np.random.permutation(num_samples)
        shuffled_texts = [texts[i] for i in permutation]
        shuffled_labels = all_labels[permutation]

        for i in range(0, num_samples, batch_size):
            batch_texts = shuffled_texts[i:i + batch_size]
            batch_labels = shuffled_labels[i:i + batch_size]

            if len(batch_texts) == 0:
                continue

            input_embeddings_batch = get_vectors_batch(batch_texts)

            input_layer_outputs, hidden_layer_outputs, hidden_layer_z_values, output_z, predicted_outputs_batch = \
                forward_pass_batch(input_embeddings_batch)

            loss = cost_function(predicted=predicted_outputs_batch, actual=batch_labels)
            total_loss += np.sum(loss)

            back_propagation_batch(input_embeddings_batch, batch_labels, predicted_outputs_batch,
                                   hidden_layer_outputs, hidden_layer_z_values, output_z, learning_rate)

        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    print("Training complete.")
