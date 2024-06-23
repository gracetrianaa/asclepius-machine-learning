const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const classes = ["Cancer", "Non-cancer"];

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const classResult = confidenceScore > 0.5 ? 0 : 1;
    const label = classes[classResult];

    let suggestion;

    if (label === "Cancer") {
      suggestion = "Segera konsultasi dengan dokter!";
    }

    if (label === "Non-cancer") {
      suggestion =
        "Pertahankan kesehatan anda dengan pola hidup sehat dan teratur.";
    }

    return { confidenceScore, label, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
  }
}

module.exports = predictClassification;
