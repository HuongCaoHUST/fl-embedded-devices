# Federated YOLO11 trên thiết bị biên với Flower

Dự án huấn luyện liên kết (Federated Learning/FL) mô hình phát hiện vật thể
Ultralytics YOLO11. Server chỉ điều phối và tổng hợp trọng số bằng FedAvg; ảnh và
nhãn luôn nằm tại từng client. Cùng một `ClientApp` có thể chạy dưới dạng client
ảo để mô phỏng hoặc chạy trên Raspberry Pi/Jetson thật.

## Kiến trúc

- `embeddedexample/task.py`: khởi tạo YOLO11, train/val và chuyển đổi trọng số.
- `embeddedexample/client_app.py`: train/đánh giá trên dữ liệu cục bộ.
- `embeddedexample/server_app.py`: FedAvg và lưu `runs/fl/final_yolo11.pt`.
- `pyproject.toml`: siêu tham số chung cho toàn bộ federation.

Chỉ tensor dấu phẩy động được tổng hợp. Các buffer số nguyên của PyTorch được
giữ cục bộ để phép lấy trung bình không làm sai kiểu dữ liệu của YOLO.

## Mô phỏng nhanh trước (2 client, COCO8)

Yêu cầu Python 3.10+ và khoảng 4 GB RAM. Tạo môi trường riêng:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
    pip install -e ".[simulation]"
```

Chạy từ thư mục dự án:

```bash
flwr run . --stream \
  --federation-config="num-supernodes=2 client-resources-num-cpus=2 client-resources-num-gpus=0"
```

Lần đầu Ultralytics sẽ tải `yolo11n.pt` làm global pretrained model ban đầu và
COCO8. Cấu hình mặc định cố ý nhỏ:
ảnh 320 px, batch 4, 1 local epoch và 2 vòng FL. COCO8 ở đây chỉ là **smoke
test**; hai client dùng chung bộ mẫu nên kết quả không đại diện cho một thí
nghiệm FL thực tế.

Nếu có một GPU và muốn mỗi lần chỉ một client dùng GPU:

```bash
flwr run . --stream \
  --run-config="device=0" \
  --federation-config="num-supernodes=2 client-resources-num-cpus=2 client-resources-num-gpus=1"
```

Sau khi chạy xong, kiểm tra checkpoint:

```bash
python -c "from ultralytics import YOLO; YOLO('runs/fl/final_yolo11.pt').val(data='coco8.yaml', imgsz=320)"
```

## Mô phỏng bằng Docker Compose: 1 server và 2 client

Yêu cầu Docker Engine và Docker Compose V2. Dự án cung cấp ba dịch vụ chạy lâu
dài (`server`, `client-1`, `client-2`) và một dịch vụ ngắn hạn `submit` dùng để
gửi Flower App lên server.

Điều khiển tự động bằng `config.yaml`:

```yaml
training:
  auto_start: true
  minimum_clients: 2
  local_epochs: 1
  server_rounds: 2
  startup_delay_seconds: 5
```

Build image rồi khởi động toàn bộ cụm:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Khi `auto_start: true`, container `submit` tự gửi một job. ServerApp sau đó chờ
đủ `minimum_clients`; khi đủ hai client, vòng train đầu tiên tự bắt đầu. Theo dõi:

- `local_epochs`: số epoch YOLO mà mỗi client chạy trong một FL round.
- `server_rounds`: tổng số vòng FL/FedAvg phía server.

```bash
docker compose logs -f submit server client-1 client-2
```

Đặt `auto_start: false` nếu chỉ muốn khởi động hạ tầng. Khi đó gửi job thủ công:

```bash
docker compose run --rm submit \
  flwr run . docker --stream
```

Checkpoint cuối được ghi ra máy host tại `runs/fl/final_yolo11.pt`. Xem log của
hạ tầng bằng:

```bash
docker compose logs -f server client-1 client-2
```

Dừng mô phỏng:

```bash
docker compose down
```

### Mô phỏng Docker Compose bằng GPU

Máy host cần NVIDIA driver và NVIDIA Container Toolkit. Kiểm tra trước:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

File `compose.gpu.yaml` đổi PyTorch sang wheel CUDA 12.6 chính thức và cấp GPU
cho ba SuperNode client. Build, xác nhận CUDA rồi chạy:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps \
  client-1 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
docker compose -f compose.yaml -f compose.gpu.yaml up
```

Theo dõi và dừng đúng project GPU:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml logs -f \
  submit server client-1 client-2 client-3
docker compose -f compose.yaml -f compose.gpu.yaml down
```

Nếu ba client dùng chung một GPU ít VRAM, giảm `batch-size` xuống 1 và
`image-size` xuống 320. Các client train đồng thời nên tổng VRAM là tổng mức sử
dụng của cả ba tiến trình.

Muốn xóa cả cache model/dataset đã tải, dùng `docker compose down -v`. Lệnh này
sẽ khiến lần chạy kế tiếp phải tải lại `yolo11n.pt` và COCO8.

Để dùng dữ liệu riêng, giữ cấu trúc `datasets/client_0` và `datasets/client_1`
như phần kế tiếp, rồi chạy job với cấu hình ghi đè:

```bash
docker compose run --rm submit \
  flwr run . docker --stream \
  --run-config="dataset-config='datasets/client_{partition_id}/data.yaml'"
```

## Mô phỏng với dữ liệu riêng cho từng client

Chuẩn bị dữ liệu theo định dạng YOLO, ví dụ:

```text
datasets/
├── client_0/
│   ├── data.yaml
│   ├── images/train/ ...
│   ├── images/val/ ...
│   ├── labels/train/ ...
│   └── labels/val/ ...
└── client_1/
    └── ...
```

Mỗi `data.yaml` dùng đường dẫn tuyệt đối hoặc đường dẫn tương đối với chính file
YAML đó:

```yaml
path: .
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

Các client phải có cùng số lớp và thứ tự `names`. Ký hiệu `{partition_id}` được
thay tự động bằng ID client ảo:

```bash
flwr run . --stream \
  --run-config="dataset-config='datasets/client_{partition_id}/data.yaml' num-server-rounds=10 local-epochs=2 image-size=640 batch-size=8" \
  --federation-config="num-supernodes=2 client-resources-num-cpus=2 client-resources-num-gpus=0"
```

Nếu thiếu RAM/VRAM, giảm `batch-size` hoặc `image-size`. Để huấn luyện từ đầu,
đổi `pretrained-model="yolo11n.yaml"`; để dùng biến thể lớn hơn, chọn `yolo11s.pt`,
`yolo11m.pt`, ... (thiết bị biên thường nên bắt đầu bằng `yolo11n.pt`).

### Global pretrained model ban đầu

`pretrained-model` là nguồn duy nhất dùng để khởi tạo global model tại server.
Nó có thể là tên checkpoint Ultralytics hoặc đường dẫn tới checkpoint riêng:

```bash
# Checkpoint chính thức, tự tải trong lần đầu
docker compose run --rm submit \
  flwr run . docker --stream \
  --run-config="pretrained-model='yolo11n.pt'"

# Hoặc checkpoint riêng: chép vào ./models (được mount read-only vào container)
docker compose run --rm submit \
  flwr run . docker --stream \
  --run-config="pretrained-model='/app/models/my_yolo11.pt'"
```

Luồng trọng số ở mỗi vòng là:

```text
pretrained checkpoint -> global model server -> client 1/client 2
                      -> local train -> FedAvg -> global model vòng kế tiếp
```

Trước mỗi lần local train, client dựng cùng kiến trúc rồi ghi đè bằng toàn bộ
global floating weights nhận từ server. Ultralytics tiếp tục train trực tiếp từ
global weights này; chúng không bị khởi tạo lại.

## Chạy trên thiết bị thật

Trên máy server, chạy SuperLink:

```bash
flower-superlink --insecure
```

Trên từng thiết bị, cài project/dependency và kết nối tới server. Mỗi thiết bị
trỏ đến YAML dữ liệu cục bộ khác nhau:

```bash
flower-supernode --insecure --superlink="SUPERLINK_IP:9092" \
  --node-config="dataset-config='/absolute/path/to/data.yaml'"
```

Thêm kết nối Control API vào cấu hình Flower trên máy điều phối:

```toml
[superlink.embedded-federation]
address = "127.0.0.1:9093"
insecure = true
```

Sau đó chạy:

```bash
flwr run . embedded-federation --stream
```

Không dùng `--insecure` ngoài mạng thử nghiệm tin cậy; khi triển khai thật cần
bật TLS và xác thực SuperNode. Hướng dẫn cài hệ điều hành cho Raspberry Pi/Jetson
nằm trong [device_setup.md](device_setup.md).

## Tham số chính

Có thể sửa `[tool.flwr.app.config]` trong `pyproject.toml` hoặc ghi đè bằng
`--run-config`:

- `pretrained-model`: checkpoint/YAML khởi tạo global model ban đầu.
- `dataset-config`: `data.yaml`, tên dataset tích hợp, hoặc mẫu có
  `{partition_id}`.
- `num-server-rounds`, `local-epochs`: số vòng server và epoch tại client.
- `fraction-train`, `fraction-evaluate`: tỷ lệ client tham gia mỗi vòng.
- `image-size`, `batch-size`, `learning-rate`, `device`.
- `output-path`: checkpoint toàn cục cuối cùng.
- `val-metrics-path`: file CSV lưu metric validation của từng node theo từng
  round (mặc định `runs/fl/val_metrics_by_node.csv`).

`min-*-nodes` mặc định là 2. Khi đổi số client, cần điều chỉnh các giá trị này để
không lớn hơn số SuperNode sẵn sàng.

### Chọn phần mô hình để FedAvg

`config.yaml` điều khiển các phần YOLO được gửi lên server, FedAvg và gửi lại
client:

```yaml
training:
  merge_parts:
    - backbone
    - head
```

Ba giá trị hợp lệ là `backbone`, `neck`, `head`. Chọn đủ cả ba tương đương
FedAvg toàn bộ model. Phần không được chọn không truyền qua mạng và được giữ
local riêng trên từng client qua các round. Với YOLO11, module 0-10 là backbone,
module 11-22 là neck và module Detect cuối (23) là head.

Legacy server và Jetson client phải dùng cùng `config.yaml`. Có thể ghi đè trực
tiếp ở cả hai phía bằng `--merge-parts backbone,head`. Log lúc khởi động sẽ in
các phần được federate và số tensor trao đổi để kiểm tra cấu hình.

## Jetson Nano với JetPack 4

JetPack 4 dùng Python 3.8 và NVIDIA PyTorch 1.11, vì vậy client Jetson dùng
Flower 1.10 classic transport thay vì ServerApp/SuperNode 1.28. Server legacy và
client Jetson phải chạy cùng phiên bản Flower 1.10.

Build và chạy server trên PC/server:

```bash
docker build -f Dockerfile.legacy-server -t fl-yolo-legacy-server .
docker run --rm -it --name fl-server -p 8080:8080 \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/datasets/client_0/data.yaml:/app/data.yaml:ro" \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/runs:/app/runs" \
  fl-yolo-legacy-server --clients 3 --rounds 2 \
  --model /app/models/yolo11n.pt --data-config /app/data.yaml
```

Build client trực tiếp trên mỗi Jetson Nano:

```bash
docker build --network host -f Dockerfile.jetson-nano \
  -t fl-yolo-jetpack4:local .
```

Chạy client 0 (đổi IP server và node/dataset trên từng Jetson):

```bash
docker run --rm -it --runtime nvidia --ipc=host --network host \
  -v "$PWD/datasets/client_0:/app/datasets/client_0:ro" \
  -v "$PWD/models:/app/models:ro" -v "$PWD/runs:/app/runs" \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  fl-yolo-jetpack4:local \
  --server 192.168.1.10:8080 --node-id jetson-0 \
  --data /app/datasets/client_0/data.yaml \
  --model /app/models/yolo11n.pt \
  --batch-size 1 --image-size 320 --workers 0 --device 0
```

Server chờ đủ `--clients` trước khi bắt đầu. Mỗi thiết bị cần `--node-id` riêng;
client 1 và 2 mount `client_1`, `client_2` tương ứng. Checkpoint toàn cục được
lưu ở `runs/fl/final_yolo11.pt`, metric từng Jetson được ghi vào
`runs/fl/val_metrics_by_node.csv`.

Client Jetson mặc định dùng `--workers 0`. Sau mỗi train/validation, client đưa
model về CPU, bỏ trainer/validator/dataloader và giải phóng CUDA cache. Với Nano
4 GB, nên bật 4 GB swap. Có thể thêm giới hạn mềm cho container để chừa RAM cho
hệ điều hành (chỉ dùng nếu Docker trên thiết bị hỗ trợ memory cgroup):

```bash
--memory=3g --memory-swap=7g
```
