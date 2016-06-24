from flask import Flask, Response, request, make_response, jsonify
from camera import Camera
import cv2
import numpy
import urllib
import math

# some initialization
app = Flask(__name__)

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(400)

index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=100)

flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
brute_force_matcher = cv2.BFMatcher()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def fetch_remote_image(url):
    r = urllib.urlopen(url)
    image = numpy.asarray(bytearray(r.read()), dtype='uint8')
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


def capture_frame():
    buff = numpy.fromstring(Camera().get_frame(), dtype=numpy.uint8)
    return cv2.imdecode(buff, 1)


def blur_image(image):
    kernel_proportion = 0.005
    kernel_w = int(2.0 * round((image.shape[1] * kernel_proportion + 1) / 2.0) - 1)
    kernel_h = int(2.0 * round((image.shape[0] * kernel_proportion + 1) / 2.0) - 1)
    return cv2.GaussianBlur(image, (kernel_w, kernel_h), 0)


def resize_image(image):
    TARGET_PIXEL_AREA = 100000.0
    ratio = float(image.shape[1]) / float(image.shape[0])
    new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
    new_w = int((new_h * ratio) + 0.5)
    return cv2.resize(image, (new_w, new_h))


@app.route('/stream')
def stream():
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/features')
def feature():
    # capture 1 frame from the camera
    index_image = capture_frame()

    # fetch remote query image
    query_image_url = request.args.get('query')
    if query_image_url is None:
        error = {
            'message': 'A query image url must be provided via the parameter: query',
            'statuscode': 400
        }
        response = make_response(jsonify(error))
        response.headers['Content-Type'] = 'application/json'
        response.status_code = 400
        return response

    query_image = fetch_remote_image(query_image_url)

    # resize query image
    query_image = resize_image(query_image)

    # lightly blur query image
    query_image = blur_image(query_image)

    # detect features of both images
    descriptor = request.args.get('descriptor')
    if descriptor == 'surf':
        kp1, des1 = surf.detectAndCompute(query_image, None)
        kp2, des2 = surf.detectAndCompute(index_image, None)

    else:
        kp1, des1 = sift.detectAndCompute(query_image, None)
        kp2, des2 = sift.detectAndCompute(index_image, None)

    # compare features of both images
    matcher = request.args.get('matcher')
    if matcher == 'flann':
        matches = flann_matcher.knnMatch(des1, des2, k=2)

    else:
        matches = brute_force_matcher.knnMatch(des1, des2, k=2)

    # apply Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    threshold = request.args.get('threshold')
    if threshold is not None and len(good) >= int(threshold):
        # find the homography and perspective transform
        src_pts = numpy.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w, _ = query_image.shape

        pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # highlight object
        index_image = cv2.polylines(index_image, [numpy.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # draw matches
    matched_image = cv2.drawMatchesKnn(query_image, kp1, index_image, kp2, good, None, flags=0)

    # encode output image as a JPEG
    _, jpeg = cv2.imencode('.jpg', matched_image)

    # return output image
    response = make_response(jpeg.tostring())
    response.headers['x-match-count'] = len(matches)
    response.headers['Content-Type'] = 'image/jpeg'
    response.headers['Content-Disposition'] = 'attachment; filename=result.jpg'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)
