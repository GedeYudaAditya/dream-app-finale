var socket = io.connect(
  window.location.protocol + "//" + document.domain + ":" + location.port,
  {
    transports: ["websocket"],
  }
);
// window.location.protocol + '//' + document.domain + ':' + location.port
socket.on("connect", function () {
  console.log("Connected...!", socket.connected);
});

var canvas = document.getElementById("canvas");
var context = canvas.getContext("2d");
const video = document.querySelector("#videoElement");

var degree = document.getElementById("degree");
var statusH = document.getElementById("status");
var statusFPS = document.getElementById("fps");
var model = document.getElementById("nama_model");
var videobox = document.getElementById("video_stream");
var btn_post = document.getElementById("btn_post");
var file = document.getElementById("file");
var isClicked = false;
var path = "";

video.width = 400; //400
video.height = 300; //300

navigator.mediaDevices
  .enumerateDevices()
  .then(function (devices) {
    const cameras = devices.filter((device) => device.kind === "videoinput");
    const camera_index = document.getElementById("camera_index");

    for (let i = 0; i < cameras.length; i++) {
      const option = document.createElement("option");
      option.value = i;
      option.text = cameras[i].label;
      camera_index.appendChild(option);
    }

    camera_index.addEventListener("change", function () {
      const selectedCamera = cameras[this.value];
      const constraints = {
        video: {
          deviceId: selectedCamera.deviceId,
        },
      };

      btn_post.addEventListener("click", function () {
        // isClicked = true;
        if (navigator.mediaDevices.getUserMedia) {
          navigator.mediaDevices
            .getUserMedia(constraints)
            .then(function (stream) {
              video.srcObject = stream;
              video.play();
            })
            .catch(function (err0r) {});
        }
      });
    });
  })
  .catch(function (error) {
    console.error("Tidak dapat mengakses perangkat: " + error);
  });

// if (navigator.mediaDevices.getUserMedia) {
//   navigator.mediaDevices
//     .getUserMedia({
//       video: true,
//     })
//     .then(function (stream) {
//       video.srcObject = stream;
//       video.play();
//     })
//     .catch(function (err0r) {});
// }

const FPS = 3;
setInterval(() => {
  width = video.width;
  height = video.height;
  context.drawImage(video, 0, 0, width, height);
  var data = canvas.toDataURL("image/jpeg", 0.5);
  context.clearRect(0, 0, width, height);
  socket.emit("image", data);

  // update degree from route /streamtext
  $.ajax({
    url: "/streamtext",
    type: "GET",
    success: function (response) {
      // console.log(response);
      degree.innerHTML = response.degree + "&deg;";
      statusFPS.innerHTML = response.mean_fps + " FPS";
      model.innerHTML = response.model;
      if (response.status) {
        degree.style.color = "green";
        statusH.innerHTML = "Muncul";
      } else {
        degree.style.color = "red";
        statusH.innerHTML = "Tidak Muncul";
      }
    },
    error: function (xhr) {
      // console.log(xhr);
    },
  });
}, 1000 / FPS);

socket.on("processed_image", function (image) {
  photo.setAttribute("src", image);
});
