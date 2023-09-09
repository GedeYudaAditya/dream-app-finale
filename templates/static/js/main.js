// make loading screen disappear
$(window).on("load", function () {
  $(".loading").fadeOut(500);
});

// ambil kamera
// $(document).ready(function () {
//   var camera_index = $("#camera_index").val();
//   // var video = document.querySelector("video");
//   // var canvas = document.querySelector("canvas");
//   // var ctx = canvas.getContext("2d");
//   // var localMediaStream = null;
//   // var onCameraFail = function (e) {
//   //   console.log("Camera did not work.", e);
//   // };

//   $("#camera_index").change(function () {
//     camera_index = $("#camera_index").val();
//     console.log(camera_index);

//     if (camera_index) {
//       $("#video_stream").attr("src", "/stream");
//     } else {
//       // replace the src of video_stream with empty
//       $("#video_stream").attr("src", "");
//     }
//   });

//   // request post to index
//   $("#btn_post").click(function () {
//     // var canvas = document.querySelector("canvas");
//     // var dataURL = canvas.toDataURL("image/png");
//     $.ajax({
//       type: "POST",
//       url: "/",
//       data: {
//         camera_index: camera_index,
//       },
//     }).done(function (o) {
//       // refresh the img with id named video_stream
//       $("#video_stream").attr("src", "/stream");
//       console.log("saved");
//     });
//   });
// });
