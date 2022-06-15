window.onload = function () {
    let imageInput = document.getElementById("image-input");
    imageInput.addEventListener("change", handleImageInput, false);
}

function handleImageInput() {
    const ctx = document.querySelector("canvas").getContext("2d");
    const [image, url] = getImageAndUrl(this.files[0])

    image.onload = () => {
        draw(image, ctx)
        const imageData = getImageData(ctx)
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

function draw(image, ctx) {
    target_size = [image.width / 5, image.height / 5];
    ctx.clearRect(0, 0, image.width, image.height);
    ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, target_size[0], target_size[1]);
}

function getImageData(ctx) {
    imageData = ctx.getImageData(0, 0, target_size[0], target_size[1]);
    return imageData;
}

function showResult(result) {
    const display = document.getElementById("prediction");
    display.innerHTML = result;
}