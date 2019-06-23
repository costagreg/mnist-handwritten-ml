const canvas = document.getElementById("mainCanvas");
const ctx = canvas.getContext("2d");

let isDrawing = false;
let prev_x, prev_y;

ctx.lineWidth = 10;
ctx.lineJoin = "round";
ctx.lineCap = "round";
ctx.strokeStyle = "black";

ctx.rect(0, 0, 280, 280);
ctx.fillStyle = "white";
ctx.fill();

canvas.addEventListener("mousedown", function(e) {
  isDrawing = true;
});

canvas.addEventListener("mousemove", function(e) {
  const x = e.clientX;
  const y = e.clientY;
  if (isDrawing) {
    if (!prev_x || !prev_y) {
      prev_x = x;
      prev_y = y;
    }

    ctx.beginPath();
    ctx.moveTo(prev_x, prev_y);
    ctx.lineTo(x, y);
    ctx.closePath();
    ctx.stroke();

    prev_x = x;
    prev_y = y;
  }
});

canvas.addEventListener("mouseup", function(e) {
  isDrawing = false;
  prev_x = null;
  prev_y = null;
  recognize();
});

const recognize = () => {
  const data = canvas.toDataURL();

  fetch("http://localhost:5002/recognize", {
    method: "POST",
    mode: "cors",
    cache: "no-cache",
    headers: {
      "Content-Type": "application/json"
    },
    redirect: "follow",
    body: JSON.stringify({ data })
  })
    .then(response => response.json())
    .then(response => {
      document.getElementById("recognized_num").innerHTML = response.number;
    });
};

const clear_canvas = () => {
  ctx.rect(0, 0, 280, 280);
  ctx.fillStyle = "white";
  ctx.fill();
  document.getElementById("recognized_num").innerHTML = "";
};
