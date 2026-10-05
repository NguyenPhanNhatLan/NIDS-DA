"""Render thesis section 3.2 from a completed Spark manifest, without training."""
import argparse
import hashlib
import json
from pathlib import Path


def render(manifest):
    if manifest.get('dataset_statistics_stage') != 'clean_after_exact_deduplication_before_split':
        raise ValueError('Rerun the updated Spark pipeline; dataset statistics are missing')
    table = ['| Dataset | Rows | Original features | Benign | Attack | Attack ratio |',
             '|---|---:|---:|---:|---:|---:|']
    inputs = []
    for domain, name in [('unsw', 'UNSW-NB15'), ('cicids', 'CICIDS2017')]:
        p = manifest['profiles'][domain]
        rows, benign, attack = p['rows'], p['benign_rows'], p['attack_rows']
        if rows <= 0 or benign + attack != rows:
            raise ValueError(f'Inconsistent counts for {domain}')
        counted = sum(r['count'] for r in manifest['counts']['class_counts'] if r['domain'] == domain)
        if counted != rows:
            raise ValueError(f'Split counts do not reconcile for {domain}')
        fmt = lambda n: f'{n:,}'.replace(',', '.')
        ratio = f'{100 * attack / rows:.4f}'.replace('.', ',') + '%'
        table.append(f'| {name} | {fmt(rows)} | {p["original_feature_count"]} | {fmt(benign)} | {fmt(attack)} | {ratio} |')
        inputs.append(f'{name}: đọc {fmt(p["input_rows"])} dòng đầu vào, loại {fmt(p["exact_duplicates_removed"])} dòng trùng, còn {fmt(rows)} dòng.')
    return '''## 3.2 Dữ liệu

Nghiên cứu sử dụng hai bộ dữ liệu lưu lượng mạng UNSW-NB15 và CICIDS2017 để xây dựng bài toán phát hiện xâm nhập liên bộ dữ liệu. Một bộ dữ liệu đóng vai trò miền nguồn và bộ còn lại là miền đích; thực nghiệm được thiết kế theo cả hai chiều chuyển miền.

### 3.2.1 UNSW-NB15

UNSW-NB15 được xây dựng tại UNSW Canberra bằng công cụ IXIA PerfectStorm, kết hợp hoạt động mạng bình thường và các hành vi tấn công trong môi trường thử nghiệm. Đặc trưng được trích xuất bằng Argus, Bro-IDS và các thuật toán bổ sung. Bộ dữ liệu bao gồm chín nhóm tấn công: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode và Worms. Trong nghiên cứu này, nhãn nhị phân gốc được sử dụng: 0 là lưu lượng bình thường, 1 là tấn công; attack_cat không được dùng làm đầu vào mô hình. [1]

### 3.2.2 CICIDS2017

CICIDS2017 do Canadian Institute for Cybersecurity xây dựng, gồm lưu lượng bình thường và nhiều kịch bản tấn công trong môi trường mạng thử nghiệm. Các đặc trưng ở mức luồng được trích xuất bằng CICFlowMeter. Để thống nhất bài toán phân loại nhị phân, nhãn BENIGN được chuyển thành 0 và các nhãn tấn công hợp lệ được chuyển thành 1. Tên loại tấn công không được đưa vào vector đặc trưng. [2]

### 3.2.3 Quy mô và phân bố lớp

Bảng dưới đây sử dụng số liệu của lần chạy Spark đã hoàn thành. Rows, Benign và Attack được thống kê sau khi loại các bản ghi trùng theo toàn bộ cột dữ liệu đã chuẩn hóa tên, trước khi chia train/validation/test. Giá trị không hợp lệ trong năm đặc trưng được chuyển thành null để xử lý ở bước tiền xử lý; không tự động xóa toàn bộ dòng. Attack ratio = Attack / Rows × 100%.

''' + '\n'.join(table) + '''

Original features là số cột đầu vào không phải nhãn sau chuẩn hóa tên cột và hợp nhất cột trùng tên tương đương, trước khi chọn đặc trưng cho mô hình; số này vẫn bao gồm các cột định danh có trong đầu vào. UNSW-NB15 loại Label và attack_cat khỏi phép đếm. Với CICIDS2017, hai cột Fwd Header Length và Fwd Header Length.1 được kiểm tra nhất quán và hợp nhất, nên chỉ tính một lần. Các số này mô tả bản CSV thực tế sử dụng, không thay thế số đặc trưng do tác giả bộ dữ liệu công bố và không phải số chiều đầu vào cuối cùng của mô hình.

''' + '\n\n'.join(inputs) + '''

Phân bố benign/attack trong bảng cho thấy mức mất cân bằng lớp của từng bộ dữ liệu. Tỷ lệ tấn công được báo cáo riêng cho mỗi miền để tránh diễn giải kết quả chỉ dựa trên accuracy. Số dòng sau làm sạch cũng cần được phân biệt với số dòng đầu vào và quy mô của các phiên bản train/test do tác giả cung cấp sẵn.

### 3.2.4 Lý do lựa chọn và khó khăn của bài toán liên bộ dữ liệu

Hai bộ dữ liệu được lựa chọn vì đều có lưu lượng bình thường, nhiều loại tấn công và nhãn phục vụ đánh giá phát hiện xâm nhập, đồng thời được xây dựng trong các môi trường khác nhau. Sự khác biệt này phù hợp với mục tiêu đánh giá khả năng chuyển mô hình từ miền nguồn sang miền đích thay vì chỉ đánh giá trên dữ liệu cùng miền. Các bản CSV ở mức luồng cũng phù hợp với pipeline xử lý bằng Spark. [1, 2]

Bài toán liên bộ dữ liệu khó do sự khác biệt về môi trường thu thập, dịch vụ mạng, thành phần tấn công, tỷ lệ lớp và cách trích xuất đặc trưng. UNSW-NB15 và CICIDS2017 sử dụng các bộ công cụ trích xuất khác nhau, nên các cột có tên hoặc ý nghĩa gần nhau chưa chắc có cùng đơn vị, công thức hay phạm vi thống kê. Vì vậy, mô hình có thể học tốt những quy luật riêng của miền nguồn nhưng không giữ được hiệu quả khi áp dụng sang miền đích. [1, 2]

Trong thiết lập proposal_v2, dữ liệu được đối chiếu về năm đặc trưng chung: flow_duration, fwd_packets, bwd_packets, fwd_bytes và bwd_bytes. Việc đồng nhất số chiều chỉ tạo điều kiện để mô hình tiếp nhận dữ liệu hai miền; không bảo đảm phân phối hoặc quan hệ giữa đặc trưng và nhãn đã giống nhau. Đây là cơ sở để nghiên cứu các phương pháp thích nghi miền và đánh giá riêng khả năng phân loại trên miền đích.

Nguồn:

[1] UNSW, The UNSW-NB15 Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset

[2] University of New Brunswick, Intrusion detection evaluation dataset CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    content = render(json.loads(raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(content)
    provenance = args.output.with_suffix('.provenance.json')
    with provenance.open('x') as stream:
        json.dump({'manifest': str(args.manifest.resolve()), 'manifest_sha256': hashlib.sha256(raw).hexdigest()}, stream, indent=2)
    print(args.output)


if __name__ == '__main__':
    main()
