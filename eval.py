import argparse
from ultralytics import YOLO

def evaluate(weights, data_yaml, imgsz=768):
    model = YOLO(weights)
    results = model.val(data=data_yaml, imgsz=imgsz, batch=32, device=0, plots=True)
    print("\n========== Evaluation Results ==========")
    print(f"mAP50:      {results.box.map50:.4f}")
    print(f"mAP50-95:   {results.box.map:.4f}")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    f1 = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-9)
    print(f"F1-score:   {f1:.4f}")
    print("=======================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', default='data/visdrone.yaml')
    parser.add_argument('--imgsz', type=int, default=768)
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.imgsz)
