from passlib.hash import sha256_crypt
import mysql.connector.pooling
import mysql.connector
import base64
import os

import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory, request, session, redirect, url_for, Response, jsonify, flash
from flask_socketio import SocketIO, emit

import requests

# Kebutuhan Untuk Prosesing Model
import cv2
import imutils
import numpy as np
import cvzone
import math
import glob
import os
from natsort import natsorted
import datetime
import pandas as pd

# for model
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import time as waktu
from step5_metricsForTraining import dice_loss, dice_coef, iou, iou_loss

# for GUI
from PIL import Image, ImageOps, ImageFont, ImageDraw

# imageHeight = 500
# imageWidth = 888
# 128, 227
# 360, 639
imageHeight = 500
imageWidth = 888
minDistance = imageHeight/6.25
# minDistance = 10

global model

# for database
# import hashlib

# Configure the connection pool
dbconfig = {
    "pool_name": "mypool",
    "pool_size": 10,
    "user": "root",
    "password": "",
    "host": "localhost",
    "database": "dolphin",
}

# default model
model_file_name = 'modelSegmenDolphin_jaccardLoss_eksperimenv1_08-11-2023_13-26-14.h5'

cnxpool = mysql.connector.pooling.MySQLConnectionPool(**dbconfig)

# connect = mysql.connector.connect(host='localhost', port='3306', user='root', password='', database='dolphin')
# cursor = connect.cursor()


app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!"
app.config["UPLOAD_FOLDER"] = "/templates/static/uploads"
socketio = SocketIO(app, async_mode="eventlet")


@app.route("/favicon.ico")
def favicon():
    """
    The favicon function serves the favicon.ico file from the static directory.

    :return: A favicon
    """
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def base64_to_image(base64_string):
    """
    The base64_to_image function accepts a base64 encoded string and returns an image.
    The function extracts the base64 binary data from the input string, decodes it, converts 
    the bytes to numpy array, and then decodes the numpy array as an image using OpenCV.

    :param base64_string: Pass the base64 encoded image string to the function
    :return: An image
    """
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@socketio.on("connect")
def test_connect():
    """
    The test_connect function is used to test the connection between the client and server.
    It sends a message to the client letting it know that it has successfully connected.

    :return: A 'connected' string
    """
    print("Connected")
    emit("my response", {"data": "Connected"})


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < minDistance:
                return True
            elif i == row1-1 and j == row2-1:
                return False


def dolphinCentroid(contours):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))
    # print(LENGTH)

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH-1:
            for j, cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i+1

    allCont = []
    maximum = int(status.max())+1
    # print(f"ini maksimum: {maximum}")
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            allCont.append(hull)

    sorted_contours = sorted(allCont, key=cv2.contourArea, reverse=True)
    largest_item = sorted_contours[0]

    hull = cv2.convexHull(largest_item)

    # find centroid
    M = cv2.moments(hull)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except ZeroDivisionError:
        cx = 0
        cy = 0
    # cv2.circle(img, (cx, cy), 7, (70, 255, 255), 7)
    # cv2.putText(img, "center", (cx - 20, cy - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 255, 255), 2)

    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    # print(f"x: {cx} y: {cy}")

    # cv2.imshow('testing', img)
    # cv2.waitKey(0)
    return cx, cy, hull


def slope(p1, p2):
    pembilang = p2[1]-p1[1]
    penyebut = p2[0]-p1[0]
    if penyebut == 0:
        result = 0
    else:
        result = pembilang / penyebut
    return (result)


def findAngle(points):
    a = points[1]
    b = points[0]
    c = points[2]
    cx = points[1][0]
    cy = points[1][1]
    droneX = points[0][0]
    droneY = points[0][1]
    m1 = slope(b, a)
    m2 = slope(b, c)
    angle = math.atan((m2 - m1)/1 + m1 * m2)
    angle = round(math.degrees(angle))

    # kuadran 1 -> gt_35.png
    if cx < droneX and cy < droneY:
        angle = angle * -1
    if cx == droneX and cy < droneY:
        angle = 90

    # kuadran 2 -> gt_2.png
    if cx > droneX and cy < droneY:
        angle = 180 - angle
    if cy == droneY and cx > droneX:
        angle = 180

    # kuadran 3 -> gt_94.png
    if cx > droneX and cy > droneY:
        angle = 180 + (angle * -1)
    if cx == droneX and cy > droneY:
        angle = 270

    # kuadran 4 -> gt_135.png
    if cx < droneX and cy > droneY:
        angle = 360 - angle
    if cy == droneY and cx < droneX:
        angle = 0

    # cv2.putText(image,str((angle)),(b[0]-100,b[1]+100), cv2.FONT_HERSHEY_DUPLEX, 2,(0,0,255),2,cv2.LINE_AA)

    # cv2.imshow('Output', image)
    return angle


def finalize(prediction, frame):
    imageOri = cv2.imread(prediction)
    # print(filename_direktoriCitra)
    # imageOri = imutils.resize(imageOri, height = imageHeight)
    imageOri = cv2.resize(imageOri, (imageWidth, imageHeight))
    image = cv2.cvtColor(imageOri, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)

    h, w, c = imageOri.shape
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

    # centroid drone
    droneX = int(w/2)
    droneY = int(h/2)
    pointDrone = [droneX, droneY]

    imageResult = imageOri

    # check if contour is available or not
    if len(contours) == 0:
        # print('kosong')

        # status kemuncul lumba-lumba -> tidak muncul = False
        status = False

        # derajat jika lumba-lumba tidak muncul
        degree = 90
    else:

        # status kemuncul lumba-lumba -> tidak muncul = False
        status = True

        # centroid dolphin
        clx, cly, largest_item = dolphinCentroid(contours)
        printCentroidDolphin = str(clx) + ', ' + str(cly)

        lx, ly, lw, lh = cv2.boundingRect(largest_item)
        cv2.rectangle(imageResult, (lx, ly), (lx+lw, ly+lh), (255, 255, 0), 2)
        # print(clx, cly)
        pointDolphin = [clx, cly]
        pointHelper = [clx, droneY]

        points = [pointDrone, pointDolphin, pointHelper]

        degree = findAngle(points)
        # print(degree)

        # shape
        cv2.circle(imageResult, (clx, cly), radius=0,
                   color=(0, 255, 0), thickness=7)
        cv2.putText(imageResult, printCentroidDolphin, (clx, cly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.arrowedLine(imageResult, (droneX, droneY),
                        (clx, cly), (255, 120, 0), 4)
        cv2.line(imageResult, (droneX, droneY),
                 (clx, droneY), (120, 200, 0), 1)
        cv2.line(imageResult, (clx, cly), (clx, droneY), (120, 200, 0), 1)

    # open drone image
    droneImage = cv2.imread('assets/drone.png', cv2.IMREAD_UNCHANGED)
    droneImage = imutils.resize(droneImage, height=90)
    droneImage = imutils.rotate_bound(droneImage, degree-90)

    # open penanda derajat image
    penandaDerajat = cv2.imread(
        'assets/penanda_derajat.png', cv2.IMREAD_UNCHANGED)

    # image Combine
    frame = cv2.imread(frame)
    frame = cv2.resize(frame, (imageWidth, imageHeight))
    hasilBlend = cv2.addWeighted(frame, 1, imageResult, 1, 0)
    cv2.imwrite("hasilBlend.png", hasilBlend)

    # #image Combine with penanda Derajat
    # hasilBlend = cv2.imread('hasilBlend.png', cv2.IMREAD_UNCHANGED)
    # hasilBlend = cv2.addWeighted(hasilBlend, 1, penandaDerajat, 1, 0)
    # cv2.imwrite("hasilBlend.png", hasilBlend)

    hDrone, wDrone, c = droneImage.shape
    hDrone = int(droneY - (hDrone/2))
    wDrone = int(droneX - (wDrone/2))
    imageResult = cvzone.overlayPNG(hasilBlend, droneImage, [wDrone, hDrone])
    imageResult = cvzone.overlayPNG(imageResult, penandaDerajat, [0, 0])

    degree_show = str(degree)
    degree_length = len(degree_show)
    # print(f"ini panjang derajat: {degree_length}")

    # kosong
    # cv2.circle(imageResult, (droneX, droneY), radius=0, color=(255, 255, 0), thickness=7)
    cv2.putText(imageResult, degree_show, (droneX - 100, droneY + 100),
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 150), 2, cv2.LINE_AA)

    # draw degree symbol
    fontpath = "assets/openSans.ttf"
    font = ImageFont.truetype(fontpath, 60)
    # fontPenanda = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(imageResult)
    draw = ImageDraw.Draw(img_pil)
    draw.text((droneX + 20 if degree_length == 3 else droneX - 15,
              droneY + 32),  "°", font=font, fill=((0, 0, 150, 0)))
    # draw.text((droneX + 20 if degree_length == 3 else droneX - 15, droneY + 32),  "°", font = font, fill = ((0,0,150,0)))
    imageResult = np.array(img_pil)

    return imageResult, degree_show, status


# time_taken = []
frame_comp = 0
total_time = 0


@socketio.on("image")
def receive_image(image):
    """
    The receive_image function takes in an image from the webcam, converts it to grayscale, and then emits
    the processed image back to the client.


    :param image: Pass the image data to the receive_image function
    :return: The image that was received from the client
    """
    # Decode the base64-encoded image data
    global hasil, degree_show, status, mean_fps, frame_comp, total_time

    citraHasilSegmentasi = None
    degree_show = 0
    status = None
    time_taken = []

    image = base64_to_image(image)

    originalImage = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    originalImage = cv2.resize(originalImage, (512, 512))

    # prepare image for U-Net input
    normalizeImage = originalImage / 255.0
    normalizeImage = normalizeImage.astype(np.float32)

    # waktu mulai
    start_time = waktu.time()

    dolphinPredict = model.predict(
        np.expand_dims(normalizeImage, axis=0))[0]
    dolphinPredict = dolphinPredict > 0.5

    # baru di tambah
    dolphinPredict = np.squeeze(dolphinPredict, axis=-1)
    dolphinPredict = np.expand_dims(dolphinPredict, axis=-1) * 255
    cv2.imwrite("segmen.png", dolphinPredict)
    cv2.imwrite("frame.png", image)

    try:
        citraHasilSegmentasi, degree_show, status = finalize(
            'segmen.png', 'frame.png')
        cv2.imwrite("templates/static/process/heading.png",
                    citraHasilSegmentasi)
        ret, buffer = cv2.imencode('.jpg', citraHasilSegmentasi)
        # citraHasilSegmentasi = buffer.tobytes()
        # make image response
        # hasil = 'static/process/heading.png'
        # Calculate FPS
        total_time += waktu.time() - start_time
        frame_comp += 1
        # time_taken.append(total_time)

        if total_time > 0.9:
            mean_fps = frame_comp
            frame_comp = 0
            total_time = 0

        # mean_time = np.mean(time_taken)
        # bulatkan ke 2 angka di belakang koma misal 3.58473 -> 3.58
        # mean_fps = round(mean_time, 2)

        # print(frame_comp)
        # print(f"FPS: {mean_fps}")

        frame_resized = cv2.resize(
            citraHasilSegmentasi, (imageWidth, imageHeight))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame_encoded = cv2.imencode(
            ".jpg", frame_resized, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data
        emit("processed_image", processed_img_data)

    except Exception as e:
        print("Error Warning :" + e)
        pass

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # frame_resized = cv2.resize(gray, (640, 360))
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
    # processed_img_data = base64.b64encode(frame_encoded).decode()
    # b64_src = "data:image/jpg;base64,"
    # processed_img_data = b64_src + processed_img_data
    # emit("processed_image", processed_img_data)


class MyObject:
    def __init__(self, degree, status, mean_fps, model=None):
        self.degree = degree
        self.status = status
        self.mean_fps = mean_fps
        self.model = model


@app.route('/streamtext')
def streamtext():
    if 'email' not in session:
        return redirect(url_for('login'))

    try:
        # response json
        data = MyObject(degree_show, status, mean_fps, model_file_name)

        # make to json format
        return jsonify(data.__dict__)
    except Exception as e:
        print(e)
        pass


@app.route('/', methods=['GET', 'POST'])
def index():
    # conn = connection()
    # cursor = conn.cursor()
    # cursor.execute('SELECT * FROM users')

    if 'email' not in session:
        return redirect(url_for('login'))

    title = 'Home'
    conn = cnxpool.get_connection()
    cursor = conn.cursor()

    user = cursor.execute(
        "SELECT * FROM users WHERE email = '" + str(session['email']) + "'")
    user = cursor.fetchall()
    nama = user[0][1]
    email = user[0][2]
    image = user[0][3]
    # all_cam = getCam()

    # if request.method == 'POST':
    #     camera_index = request.form['camera_index']

    #     global my_cap
    #     my_cap = cv2.VideoCapture(int(camera_index))

    #     return redirect(url_for('index'))
    cursor.close()
    conn.close()

    return render_template('index.html', title=title, nama=nama, email=email, image=image)


@app.route('/setting')
def setting():
    if 'email' not in session:
        return redirect(url_for('login'))

    title = 'Setting'
    conn = cnxpool.get_connection()
    cursor = conn.cursor()

    user = cursor.execute(
        "SELECT * FROM users WHERE email = '" + str(session['email']) + "'")
    user = cursor.fetchall()
    nama = user[0][1]
    email = user[0][2]
    image = user[0][3]

    cursor.close()
    conn.close()

    return render_template('setting.html', title=title, nama=nama, email=email, image=image)


@app.route('/setting', methods=['POST'])
def settingProses():
    if 'email' not in session:
        return redirect(url_for('login'))
    conn = cnxpool.get_connection()
    cursor = conn.cursor()
    nama = request.form['nama']
    email = request.form['email']
    password = request.form['password']
    password_confirm = request.form['password-confirmation']
    my_image = request.files['image']

    user = cursor.execute(
        "SELECT * FROM users WHERE email = '" + str(session['email']) + "'")
    user = cursor.fetchall()

    errors = []

    # validasi input
    if nama == '':
        errors.append('Nama tidak boleh kosong')

    if email == '':
        errors.append('Email tidak boleh kosong')

    if password == '':
        errors.append('Password tidak boleh kosong')

    if password_confirm == '':
        errors.append('Konfirmasi Password tidak boleh kosong')

    if password != password_confirm:
        errors.append('Konfirmasi Password tidak sama')

    if (password == password_confirm and len(errors) == 0):
        try:
            print('Ini dia')
            print(my_image.filename != '')
            if my_image.filename != '':
                # save image and rename image name
                extension = my_image.filename.split('.')[1]
                original_image_name = my_image.filename.split('.')[0]
                image_name = original_image_name + \
                    str(waktu.time()) + '.' + extension
                my_image.filename = image_name
                my_image.save('templates/static/uploads/' + my_image.filename)

                # delete old image
                # check if in database image is not null
                if user[0][3]:
                    # delete old image
                    os.remove('templates/static/uploads/' + user[0][3])
            else:
                image_name = user[0][3]

            password = sha256_crypt.encrypt(password)
            cursor.execute("UPDATE users SET nama = '" + str(nama) + "', password = '" + str(
                password) + "', image = '" + str(image_name) + "' WHERE email = '" + str(email) + "'")
            conn.commit()
            flash('Data berhasil diubah', 'success')
            cursor.close()
            conn.close()
            return redirect(url_for('index'))
        except Exception as e:
            flash('Terjadi kesalahan pada database. ' + str(e), 'error')
            print(e)
            return redirect(url_for('setting'))
    else:
        error_massage = "<div class='text-danger'>"

        for error in errors:
            error_massage += error + "<br>"

        error_massage += "</div>"
        flash("<b>Terjadi kesalahan</b>. " + error_massage, 'error')
        return redirect(url_for('setting'))


@app.route('/login')
def login():
    if 'email' in session:
        return redirect(url_for('index'))

    title = 'Login'

    return render_template('login.html', title=title)


@app.route('/login', methods=['POST'])
def loginProses():
    conn = cnxpool.get_connection()
    cursor = conn.cursor()

    if 'email' in session:
        return redirect(url_for('index'))

    email = request.form['email']
    password = request.form['password']

    # enkripsi password

    # decode password
    # password = password.encode('utf-8')

    # response = requests.post('https://reqres.in/api/login',
    #                          json={'email': email, 'password': password})

    cursor.execute("SELECT * FROM users WHERE email = '" + str(email) + "'")

    # token generate
    token = email
    token = token.encode('utf-8')

    data = cursor.fetchall()

    if data:
        condition = sha256_crypt.verify(password, data[0][4])
        # close connection here
        cursor.close()
        conn.close()
        if condition:
            session['email'] = email
            session['success'] = 'Berhasil login'
            session['nama'] = data[0][1]
            # session['token'] = token

            flash('Berhasil login', 'success')
            return redirect(url_for('index'))
        else:
            # session['error'] = 'Email atau password salah'
            flash('Email atau password salah', 'error')
            return redirect(url_for('login'))
    else:
        # session['error'] = 'Data tidak ditemukan'
        flash('Data tidak ditemukan', 'error')
        return redirect(url_for('login'))
    # Close the cursor and return the connection to the pool


@app.route('/register')
def register():
    if 'email' in session:
        return redirect(url_for('index'))

    title = 'Register'

    return render_template('register.html', title=title)


@app.route('/register', methods=['POST'])
def registerProses():
    conn = cnxpool.get_connection()
    cursor = conn.cursor()

    if 'email' in session:
        return redirect(url_for('index'))

    nama = request.form['nama']
    email = request.form['email']
    password = request.form['password']
    password_confirm = request.form['password-confirmation']

    errors = []
    # validasi input
    if nama == '':
        errors.append('Nama tidak boleh kosong')

    if email == '':
        errors.append('Email tidak boleh kosong')

    if password == '':
        errors.append('Password tidak boleh kosong')

    if password_confirm == '':
        errors.append('Konfirmasi Password tidak boleh kosong')

    if password != password_confirm:
        errors.append('Konfirmasi Password tidak sama')

    if (password == password_confirm and len(errors) == 0):
        try:
            # enkripsi password
            password = sha256_crypt.encrypt(password)
            # transaksi ke database
            cursor.execute("INSERT INTO users (nama, email, password) VALUES (%s, %s, %s)",
                           (nama, email, password))

            conn.commit()

            result = cursor.rowcount
            cursor.close()
            conn.close()

            if result > 0:
                flash('Data berhasil disimpan', 'success')
                return redirect(url_for('login'))
            else:
                flash('Data gagal disimpan', 'error')
                return redirect(url_for('register'))
        except Exception as e:
            flash('Terjadi kesalahan pada database. ' + str(e), 'error')
            print(e)
            return redirect(url_for('register'))
    else:
        error_massage = "<div class='text-danger'>"

        for error in errors:
            error_massage += error + "<br>"

        error_massage += "</div>"
        flash("<b>Terjadi kesalahan</b><br>. " + error_massage, 'error')
        return redirect(url_for('register'))


@app.route('/logout')
def logout():

    if 'email' not in session:
        return redirect(url_for('login'))

    session.pop('email', None)
    session.pop('nama', None)
    session.pop('token', None)

    return redirect(url_for('index'))


class UploadModel:
    def __init__(self, model_name, date, time):
        self.model_name = model_name
        self.date = date
        self.time = time


@app.route('/upload', methods=['POST'])
def upload():
    try:
        # for upload user model
        if 'email' not in session:
            return redirect(url_for('login'))

        conn = cnxpool.get_connection()
        cursor = conn.cursor()

        user = cursor.execute(
            "SELECT * FROM users WHERE email = '" + str(session['email']) + "'")
        user = cursor.fetchall()

        models = user[0][5]

        if models:
            # check if in static folder model there is the same file model
            if os.path.exists('templates/static/model/' + models):
                # delete old model
                os.remove('templates/static/model/' + models)

        # get model from user
        my_model = request.files['model']

        # save model and rename model name
        extension = my_model.filename.split('.')[1]
        original_model_name = my_model.filename.split('.')[0]
        model_name = original_model_name + str(waktu.time()) + '.' + extension
        my_model.filename = model_name
        my_model.save('templates/static/model/' + my_model.filename)

        # update model name in database
        cursor.execute("UPDATE users SET user_model = '" + str(model_name) +
                       "' WHERE email = '" + str(session['email']) + "'")
        conn.commit()
        cursor.close()
        conn.close()

        # store model name in global variable
        global model_file_name
        model_file_name = model_name

        # reload model
        global model
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss, 'iou_loss': iou_loss}):
            model = tf.keras.models.load_model(
                "templates/static/model/" + model_file_name)

        data = UploadModel(model_name, str(datetime.date.today()), str(
            datetime.datetime.now().strftime("%H:%M:%S")))
        return jsonify(data.__dict__)
    except Exception as e:
        return jsonify({'errors': str(e)})


if __name__ == "__main__":
    # load model from user database

    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss, 'iou_loss': iou_loss}):
        model = tf.keras.models.load_model(
            "templates/static/model/" + model_file_name)

        # "assets/modelSegmenDolphin_jaccardLoss_eksperimenv1_08-11-2023_13-26-14.h5") #terbaru experimental
        # "assets/modelSegmenDolphin_dice-loss_DATASET_AUG_500epoch_JaccardLoss_08-02-2023_20-54-23.h5") #lama
    socketio.run(app, debug=True, port=5000, host='127.0.0.1')
    #  keyfile="ssl/key.pem", certfile="ssl/cert.pem")
    # socketio.run(app, debug=True, port=5000, host='0.0.0.0') # port=5000, host='0.0.0.0' ssl_context=("ssl/laragon.crt", "ssl/laragon.key")
    # socketio.run(app, debug=True, port=5000, host='0.0.0.0', keyfile="ssl/laragon.key", certfile="ssl/laragon.crt") # port=5000, host='0.0.0.0' ssl_context=("ssl/laragon.crt", "ssl/laragon.key")
