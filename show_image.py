import cv2, os, json
import numpy as np

if __name__ == '__main__':

    # class_names = ['banana', 'orange', 'carrot', 'horse', 'bottle', 'broccoli', 'bus', 'cup', 'toothpaste', 'tie', 'wine glass']
    class_names = ['toothpaste', 'raincoat', 'tissue', 'instant-noodles', 'guai-guai', 
               'oreo', 'energy-drink', 'washing-powder', 'coca-cola', 'green-tea']
    save_path = "output_onnx_opt_10_torch18/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    with open('json/output_onnx_opt_10_torch18.json', 'r') as file:
        eval_results = json.load(file)

    for result in eval_results:
        print(result['img_path'])
        filename = os.path.basename(result['img_path'])
        img_path = os.path.join("/mnt/dataset2/test", filename)
        img = cv2.imread(img_path)
        print(img_path)
        print(img.shape)

        bboxes = result['bbox'][:4]

        for bbox in bboxes:
            x1, y1, w, h = bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
            # print(x1, y1, x2, y2)
            conf = np.round(bbox[4], 4)
            label_idx =  int(bbox[5]-1)

            cv2.rectangle(
                img, (x1, y1), (x2, y2), (255, 255, 255), 3
            )
            message = f'{class_names[label_idx]}, conf : {conf}'
            cv2.putText(
                img, message, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3
            )

        cv2.imwrite(f'{save_path}{filename}', img)

        # cv2.imshow("YOLOv8 Detection", img)
        # ## Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break



