from predictor import predict
from explainability import generate_heatmap


def analyze_image(image_path):

    result = predict(image_path)

    if result["prediction"] != "UNKNOWN":

        heatmap_path = generate_heatmap(
            image_path,
            result["prediction"]
        )

        result["heatmap"] = heatmap_path

    else:

        result["heatmap"] = None

    return result


if __name__ == "__main__":

    image_path = input("Image Path: ")

    result = analyze_image(image_path)

    print("\nRESULT:\n")
    print(result)