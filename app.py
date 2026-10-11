from flask import Flask, render_template, request
from PIL import Image
import numpy as np

app = Flask(__name__)

def get_features(file):

    image = Image.open(file).convert("L")
    image = image.resize((16, 16))

    array = np.array(image, dtype=float) / 255.0

    return array.flatten()


def normalize(vector):

    length = np.linalg.norm(vector)

    if length == 0:
        return vector

    return vector / length


def to_binary(vector):

    middle = np.mean(vector)

    result = np.where(vector >= middle, 1, -1)

    return result


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/perceptron", methods=["GET", "POST"])
def perceptron():

    result = None

    if request.method == "POST":

        class1_files = request.files.getlist("class1")
        class2_files = request.files.getlist("class2")
        test_file = request.files.get("test")

        learning_rate = float(
            request.form.get("learning_rate", 0.1)
        )

        epochs = int(
            request.form.get("epochs", 20)
        )

        X = []
        D = []

        for file in class1_files:
            if file.filename:
                vector = get_features(file)
                vector = normalize(vector)

                X.append(vector)
                D.append(-1)

        for file in class2_files:
            if file.filename:
                vector = get_features(file)
                vector = normalize(vector)

                X.append(vector)
                D.append(1)

        if len(X) == 0:
            return render_template(
                "perceptron.html",
                error="Додайте навчальні зображення."
            )

        X = np.array(X)
        D = np.array(D)

        np.random.seed(1)

        weights = np.random.uniform(
            -0.1,
            0.1,
            X.shape[1]
        )

        bias = 0.0

        history = []

        converged = False

        for epoch in range(epochs):

            errors = 0

            for i in range(len(X)):

                x = X[i]
                desired = D[i]

                s = np.dot(weights, x)

                if s > bias:
                    output = 1
                else:
                    output = -1

                error = desired - output

                if error != 0:

                    weights = (
                        weights
                        + learning_rate * error * x
                    )

                    bias = (
                        bias
                        - learning_rate * error
                    )

                    errors += 1

                history.append({
                    "epoch": epoch + 1,
                    "sample": i + 1,
                    "desired": int(desired),
                    "output": int(output),
                    "error": int(error)
                })

            if errors == 0:
                converged = True
                break

        test_vector = None
        prediction = None

        if test_file and test_file.filename:

            test_vector = get_features(test_file)

            test_vector = normalize(test_vector)

            s = np.dot(weights, test_vector)

            if s > bias:
                prediction = 1
            else:
                prediction = -1

        result = {
            "weights": weights[:20],
            "bias": bias,
            "prediction": prediction,
            "epochs": epoch + 1,
            "converged": converged,
            "history": history,
            "test_vector": (
                test_vector[:20]
                if test_vector is not None
                else None
            )
        }

    return render_template(
        "perceptron.html",
        result=result
    )

@app.route("/hopfield", methods=["GET", "POST"])
def hopfield():

    result = None

    if request.method == "POST":

        files = request.files.getlist("patterns")
        test_file = request.files.get("test")

        files = [
            file
            for file in files
            if file.filename
        ]

        if len(files) != 3:
            return render_template(
                "hopfield.html",
                error="Потрібно завантажити рівно 3 еталонні зображення."
            )

        if not test_file or not test_file.filename:
            return render_template(
                "hopfield.html",
                error="Додайте тестове зображення."
            )
            
        patterns = []

        for file in files:

            vector = get_features(file)

            vector = to_binary(vector)

            patterns.append(vector)

        patterns = np.array(patterns)

        n = patterns.shape[1]

        W = np.zeros((n, n))

        for pattern in patterns:

            W += np.outer(
                pattern,
                pattern
            )

        W = W / n

        np.fill_diagonal(W, 0)

        test_vector = get_features(test_file)

        test_vector = to_binary(test_vector)

        y = test_vector.copy()

        iterations = 0

        for i in range(30):

            new_y = np.where(
                np.dot(W, y) >= 0,
                1,
                -1
            )

            iterations += 1

            if np.array_equal(new_y, y):
                break

            y = new_y

        similarity = []

        for pattern in patterns:

            value = np.dot(
                y,
                pattern
            )

            similarity.append(int(value))

        class_number = (
            np.argmax(similarity) + 1
        )

        result = {
            "class": class_number,
            "iterations": iterations,
            "similarity": similarity,
            "matrix_size": W.shape,
            "output": y[:20]
        }

    return render_template(
        "hopfield.html",
        result=result
    )


@app.route("/hamming", methods=["GET", "POST"])
def hamming():

    result = None

    if request.method == "POST":

        files = request.files.getlist("patterns")
        test_file = request.files.get("test")

        files = [
            file
            for file in files
            if file.filename
        ]

        if len(files) != 3:
            return render_template(
                "hamming.html",
                error="Потрібно завантажити рівно 3 еталонні зображення."
            )

        if not test_file or not test_file.filename:
            return render_template(
                "hamming.html",
                error="Додайте тестове зображення."
            )

        patterns = []

        for file in files:

            vector = get_features(file)

            vector = normalize(vector)

            patterns.append(vector)

        patterns = np.array(patterns)

        n = patterns.shape[1]
        m = patterns.shape[0]

        W = patterns.T / 2

        bias = n / 2

        test_vector = get_features(test_file)

        test_vector = normalize(test_vector)

        first_layer = (
            np.dot(W.T, test_vector)
            + bias
        )

        epsilon = 1 / (m + 1)

        second_layer = first_layer.copy()

        for _ in range(30):

            new_layer = np.zeros(m)

            for j in range(m):

                other_sum = (
                    np.sum(second_layer)
                    - second_layer[j]
                )

                value = (
                    second_layer[j]
                    - epsilon * other_sum
                )

                new_layer[j] = max(
                    0,
                    value
                )

            if np.allclose(
                new_layer,
                second_layer
            ):
                break

            second_layer = new_layer


        class_number = (
            np.argmax(second_layer) + 1
        )

        result = {
            "class": class_number,
            "first_layer": first_layer,
            "second_layer": second_layer,
            "epsilon": epsilon,
            "weights_size": W.shape,
            "test_vector": test_vector[:20]
        }

    return render_template(
        "hamming.html",
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)