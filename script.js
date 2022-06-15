const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./k-on.onnx")
window.onload = function () {
    const imageInput = document.getElementById("image-input");
    imageInput.addEventListener("change", handleImageInput, false);
}

function handleImageInput() {
    const canvas = document.querySelector("canvas");
    const [image, url] = getImageAndUrl(this.files[0])

    image.onload = () => {
        draw(image, canvas, 256)
        const imageData = getImageData(canvas)
        window.URL.revokeObjectURL(url);
        console.log(imageData);
        loadingModelPromise.then(() => {
            predictCharacter(imageData, sess).then(
                (character) => { showResult(character); }
            );
        })
    }
}

async function predictCharacter(imageData, sess) {
    data = new Float32Array(extract(imageData.data));

    const input = new onnx.Tensor(data, "float32", [1, 3, 224, 224]);
    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;
    const maxPrediction = predictions.indexOf(Math.max(...predictions));

    characters = ['azusa', 'mio', 'ritsu', 'sawako', 'tsumugi', 'ui', 'yui'];

    return character[maxPrediction];
}

function extract(data) {
    // const new_data = [];
    // for (channel = 0; channel < 3; channel++) {
    //     for (idx = channel; idx < 224 * 224 * 4; idx + 4) {
    //         new_data.push(data[idx]);
    //     }
    // }
    // return new_data;
    return data.slice(224 * 224);
}

function getImageAndUrl(file) {
    const image = new Image()
    const url = window.URL.createObjectURL(file)
    image.src = url
    return [image, url]
}

function draw(image, canvas, resize) {
    const ctx = canvas.getContext('2d');
    let max_side = Math.max(image.width, image.height);
    let min_side = Math.min(image.width, image.height);
    max_side *= resize / min_side;
    min_side = resize;
    target_size = image.width > image.height ? [max_side, min_side] : [min_side, max_side]
    console.log(target_size);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = target_size[0];
    canvas.height = target_size[1];
    ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, target_size[0], target_size[1]);
}

function getImageData(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData((canvas.width - 224) / 2, (canvas.width - 224) / 2, 224, 224)
    return imageData;
}

function showResult(result) {
    const display = document.getElementById("prediction");
    display.innerHTML = result;
}