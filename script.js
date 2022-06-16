const sessionPromise = ort.InferenceSession.create('./k-on.onnx');
let session;

window.onload = function () {
    const imageInput = document.getElementById("image-input");
    imageInput.addEventListener("change", handleImageInput, false);
    sessionPromise.then((r) => { session = r; })
}

function handleImageInput() {
    const canvas = document.querySelector("canvas");
    const [image, url] = getImageAndUrl(this.files[0])

    image.onload = () => {
        clearResult();
        draw(image, canvas, 256);
        const imageData = getImageData(canvas);
        window.URL.revokeObjectURL(url);
        predictCharacter(imageData);
    }
}

async function predictCharacter(imageData) {
    data = new Float32Array(extract(imageData.data));

    const input = new ort.Tensor("float32", data, [1, 3, 224, 224]);

    const outputMap = await session.run({ 'input': input });
    const prediction = outputMap.output.data;
    const maxPrediction = prediction.indexOf(Math.max(...prediction));

    characters = ['azusa', 'mio', 'ritsu', 'sawako', 'tsumugi', 'ui', 'yui'];

    showResult(characters[maxPrediction]);
}

function extract(data) {
    const redArray = data.filter(function (v, i, a) { return i % 4 == 0; })
    const greenArray = data.filter(function (v, i, a) { return i % 4 == 1; })
    const blueArray = data.filter(function (v, i, a) { return i % 4 == 2; })
    return [...redArray, ...greenArray, ...blueArray].map(function (i) { return i / 255; });
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

function clearResult() {
    const display = document.getElementById("prediction");
    display.innerHTML = '';
}