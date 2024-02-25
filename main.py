import cv2 as cv
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
names = {0: 'person'}


def draw_rect(img, bbox):
    cv.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=2)


def write_txt(img, label, bbox):
    cv.putText(img, label, (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)


def detect(mode):
    if mode == 1:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture('./videos/people.mp4')
    print('Чтобы остановить выполнение программы нажмите клавишу "q"')
    while True:
        success, img = cap.read()
        output = model(img)
        pred = output.pred[0].tolist()
        for data in pred:
            if int(data[5]) in names.keys():  # выделяем только людей
                bbox = data[:4]  # координаты прямоугольника
                label_prob = names[data[
                    5]] + f'{round(data[4], 2)}'  # название класса обнаруженного объекта и вероятность его обнаружения
                draw_rect(img, bbox)  # рисуем прямоугольники
                write_coord = bbox[:2]
                write_txt(img, label_prob, write_coord)
        cv.imshow('Result', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    print("Введите режим тестирования: 1 - захват вебкамеры, 2 - тестовое видео")
    mode = int(input())
    assert mode in [1, 2], "Режим тестирования должен быть равен 1 или 2"
    detect(mode)
