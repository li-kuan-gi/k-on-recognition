window.onload = function () {
    let imageInput = document.getElementById("image-input");
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
        // character = predictCharacter(imageData)
        // showResult(character)
    }
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
    const imageData = ctx.getImageData((canvas.width-224)/2, (canvas.width-224)/2, 224, 224)
    return imageData;
}

function showResult(result) {
    const display = document.getElementById("prediction");
    display.innerHTML = result;
}