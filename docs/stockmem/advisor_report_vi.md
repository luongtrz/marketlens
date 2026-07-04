# Báo Cáo StockMem: Ý Tưởng, Luồng Dữ Liệu, Cách Thực Hiện Và Kết Quả

Tài liệu này tóm tắt phần StockMem trong hệ thống MarketLens để gửi giảng viên
hướng dẫn. Nội dung tập trung vào bốn điểm: mục tiêu bài toán, ý tưởng kỹ
thuật, luồng dữ liệu triển khai, và kết quả thực nghiệm hiện tại.

## 1. Tóm Tắt

StockMem là một tầng **bộ nhớ lịch sử có cấu trúc** cho bài toán dự đoán hướng
đi của thị trường crypto. Thay vì chỉ đưa dữ liệu hiện tại vào mô hình ngôn ngữ
và yêu cầu mô hình tự suy luận, StockMem lưu lại các trạng thái thị trường trong
quá khứ, truy xuất các trạng thái tương tự với ngày hiện tại, sau đó dùng kết
quả thực tế sau 7 ngày của các trạng thái lịch sử này làm bằng chứng cho quyết
định hiện tại.

Bài toán dự đoán được chuẩn hóa thành nhãn D7:

```text
BUY  nếu future_return_7d > +2%
SELL nếu future_return_7d < -2%
HOLD nếu -2% <= future_return_7d <= +2%
```

Kết quả chính trên tập test `2025-07-01` đến `2026-05-01`:

| Asset | Retriever chính | Decision head | Số dòng test | Overall DA | Active DA | Coverage | BUY DA | SELL DA | Majority@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | `learned_recency_50_50` | `count_vote_buy3_sell4` | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.7114 | 0.5443 |
| ETH | `eth_learned_recency_50_50_h30` | `mean_learned_weights_buy0.50_sell0.75` | 305 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.6496 | 0.5246 |

Kết luận chính: StockMem hoạt động như một pipeline truy xuất bằng chứng lịch
sử có cấu trúc. Hệ thống cho phép kiểm tra từng prediction thông qua top-10
evidence records, có cơ chế chống rò rỉ dữ liệu tương lai, và có thể mở rộng từ
BTC sang ETH bằng cách tạo profile riêng cho từng asset.

## 2. Bối Cảnh Và Vấn Đề

Trong dự đoán tài chính, việc chỉ dùng ngữ cảnh hiện tại thường chưa đủ vì thị
trường có tính chu kỳ, tính regime, và chịu ảnh hưởng từ các mẫu hình lịch sử.
Nếu chỉ truyền dữ liệu hiện tại cho mô hình ngôn ngữ, hệ thống có một số hạn
chế:

1. Không có cơ chế truy xuất lịch sử rõ ràng.
2. Khó giải thích vì sao một tín hiệu được đưa ra.
3. Khó kiểm soát mô hình đã dựa vào bằng chứng nào.
4. Dễ thiên lệch theo tin tức hoặc trạng thái ngắn hạn hiện tại.

StockMem được thiết kế để bổ sung một tầng historical memory. Mỗi ngày lịch sử
được lưu thành một record có cấu trúc. Khi cần dự đoán ngày hiện tại, hệ thống
tìm các record lịch sử tương tự và sử dụng kết quả D7 đã biết của các record
đó làm bằng chứng.

## 3. Ý Tưởng Kỹ Thuật

StockMem có thể được xem là một dạng Retrieval-Augmented Reasoning cho dữ liệu
thị trường. Điểm khác biệt là hệ thống không truy xuất văn bản tự do, mà truy
xuất các trạng thái thị trường được vector hóa theo nhiều nhóm đặc trưng.

Luồng ý tưởng:

```text
trạng thái thị trường hiện tại
  -> truy xuất các trạng thái lịch sử tương tự
  -> lấy outcome D7 của các trạng thái lịch sử
  -> tạo tập bằng chứng top-k
  -> decision head tổng hợp bằng chứng
  -> dự đoán BUY/HOLD/SELL
```

Pipeline được tách thành hai thành phần chính:

| Thành phần | Vai trò |
| --- | --- |
| Retriever | Tìm các bản ghi lịch sử phù hợp với query hiện tại. |
| Decision head | Chuyển tập evidence thành tín hiệu BUY/HOLD/SELL. |

Cách tách này giúp dễ đánh giá: nếu top-k evidence chưa nhất quán, vấn đề nằm ở
retriever; nếu evidence tốt nhưng prediction chưa tốt, vấn đề nằm ở decision
head.

## 4. Luồng Dữ Liệu Tổng Thể

Luồng dữ liệu của StockMem:

```text
raw market/news/factor data
  -> StockMemRecord
  -> vector blocks
  -> historical candidate pool
  -> retriever scoring
  -> top-10 evidence records
  -> decision head
  -> BUY/HOLD/SELL prediction
  -> evaluation metrics
  -> report artifacts
```

Luồng hiện tại cho BTC:

```text
data/exports/stockmem_records.ndjson
  -> stockmem/config/learned_retriever_finbert.json
  -> stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
  -> top-10 evidence
  -> count_vote_buy3_sell4
  -> BTC prediction/evaluation
```

Luồng hiện tại cho ETH:

```text
data/exports/stockmem_records_eth.ndjson
  -> stockmem/config/learned_retriever_finbert.eth.json
  -> stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
  -> top-10 evidence
  -> mean_learned_weights_buy0.50_sell0.75
  -> ETH prediction/evaluation
```

File router cho nhiều asset:

```text
stockmem/config/model_profiles.json
```

File này quy định mỗi asset dùng dataset, learned retriever artifact, retriever
config, và decision head nào. Nhờ đó BTC và ETH có thể dùng chung kiến trúc
nhưng khác profile.

## 5. Cấu Trúc Một StockMem Record

Một record trong StockMem gồm các nhóm dữ liệu chính:

| Thành phần | Ý nghĩa | Công dụng |
| --- | --- | --- |
| `date`, `symbol` | Ngày và asset. | Lọc đúng thị trường và đúng thời gian. |
| `market_snapshot` | Trạng thái thị trường tại ngày đó. | Nguồn cho indicator, price, regime. |
| `event_vec` | Vector sự kiện/tin tức. | Hỗ trợ learned retriever hiểu bối cảnh sự kiện. |
| `factor_vec` | Vector factor và taxonomy. | So sánh bối cảnh thị trường/factor. |
| `indicator_vec` | Chỉ báo kỹ thuật/sentiment dạng rút gọn. | So sánh trạng thái kỹ thuật. |
| `price_vec` | Biến động giá, volume, range. | So sánh market state và xu hướng. |
| `future_return_7d` | Return thực tế sau 7 ngày. | Tạo nhãn BUY/HOLD/SELL cho đánh giá và evidence lịch sử. |

Các file liên quan:

| File | Công dụng |
| --- | --- |
| `stockmem/src/models.py` | Định nghĩa `StockMemRecord`, `SimilarRecord` và các kiểu dữ liệu chính. |
| `stockmem/src/search/embedder.py` | Chuyển record thành các vector block. |
| `stockmem/src/search/event_memory.py` | Tạo trạng thái event theo ngày khi cần. |
| `stockmem/src/search/taxonomy.py` | Quy ước taxonomy/factor cho đặc trưng sự kiện. |

## 6. Cách Truy Xuất Evidence

### 6.1 Fixed kNN

Phiên bản nền tảng dùng weighted cosine similarity trên ba block:

```text
score(q,c) =
  w_factor    * cos(factor_q, factor_c)
+ w_indicator * cos(indicator_q, indicator_c)
+ w_price     * cos(price_q, price_c)
```

BTC fixed weights:

```text
w_factor    = 0.5443920554
w_indicator = 0.3090805325
w_price     = 0.1415662727
```

ETH fixed weights sau tuning:

```text
w_factor    = 0.3488
w_indicator = 0.2885
w_price     = 0.3627
```

Các file liên quan:

| File | Công dụng |
| --- | --- |
| `stockmem/config/weights.auto.json` | Fixed-kNN weights cho BTC. |
| `stockmem/config/weights.eth.auto.json` | Fixed-kNN weights cho ETH. |
| `stockmem/src/search/searcher.py` | Logic search runtime. |
| `stockmem/scripts/optimize_weights.py` | Script tune fixed weights. |

Fixed kNN có ưu điểm là ổn định và dễ giải thích. Cùng query, cùng candidate
pool và cùng weights sẽ cho cùng ranking.

### 6.2 Learned Retriever

Learned retriever học một metric thay vì chỉ dùng trọng số thủ công. Công thức:

```text
score(q,c) = sum_b alpha_b * cos(D_b q_b, D_b c_b)
```

Trong đó:

```text
D_b     = trọng số từng chiều trong block b
alpha_b = trọng số của block b
```

Các file chính:

| File | Công dụng |
| --- | --- |
| `stockmem/src/search/learned_metric.py` | Load và tính điểm learned diagonal metric. |
| `stockmem/config/learned_retriever_finbert.json` | Learned retriever artifact cho BTC. |
| `stockmem/config/learned_retriever_finbert.eth.json` | Learned retriever artifact cho ETH. |
| `stockmem/scripts/train_learned_retriever.py` | Train learned retriever. |
| `stockmem/scripts/retrain_finbert_retriever.py` | Fine-tune retriever từ NDJSON; dùng cho ETH. |

Sau khi artifact đã được tạo, learned retriever cũng deterministic. Nếu chạy
cùng input và cùng artifact nhiều lần, ranking không thay đổi.

### 6.3 Learned + Recency Retriever

Kết quả thực nghiệm cho thấy pure learned retrieval chưa đủ mạnh. Lý do là thị
trường có tính liên tục theo thời gian; các ngày gần hiện tại thường chứa
thông tin regime/trend quan trọng. Vì vậy pipeline hiện tại dùng learned
similarity kết hợp recency:

```text
score(q,c) =
  w_learned * learned_similarity(q,c)
+ w_recency * exp(-age_days(q,c) / half_life_days)
```

BTC:

```text
0.5 * learned_similarity + 0.5 * recency(half_life=21d)
```

ETH:

```text
0.5 * ETH learned_similarity + 0.5 * recency(half_life=30d)
```

Config chính:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
```

Ý nghĩa: retriever vừa giữ khả năng tìm mẫu hình lịch sử, vừa có nhận thức về
xu hướng gần đây.

## 7. Decision Head

Retriever chỉ trả về tập evidence. Decision head chuyển evidence thành tín
hiệu.

BTC dùng head:

```text
count_vote_buy3_sell4
```

Luật:

```text
SELL nếu sell_count >= 4 và sell_count >= buy_count
BUY  nếu buy_count  >= 3 và buy_count  >  sell_count
HOLD nếu không thỏa hai điều kiện trên
```

ETH dùng head:

```text
mean_learned_weights_buy0.50_sell0.75
```

File chọn/tune head:

```text
stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py
```

Các head được chọn trên validation split, sau đó mới báo cáo kết quả trên test
split.

## 8. Kiểm Soát Rò Rỉ Dữ Liệu Tương Lai

Để tránh dùng thông tin tương lai, khi dự đoán ngày `t`, hệ thống chỉ được dùng
record lịch sử mà kết quả D7 đã biết tại thời điểm `t`.

Quy tắc matured pool:

```text
candidate.date < query.date
candidate.date + 7 days <= query.date
```

Nếu candidate quá gần query, future_return_7d của candidate chưa thể biết tại
thời điểm query, nên candidate đó không được dùng làm evidence.

File xử lý split và matured pool:

```text
stockmem/scripts/ndjson_eval_common.py
```

Split chính:

```text
train:      đến 2024-12-24
validation: 2025-01-01 đến 2025-06-23
test:       2025-07-01 đến 2026-05-01
```

Tập validation có 174 dòng. Tập test có 305 dòng.

## 9. Các Script Và Artifact Chính

Script chính:

| Script | Chức năng |
| --- | --- |
| `stockmem/scripts/evaluate_stockmem_strict_models.py` | So sánh strict fixed/learned retriever và head. |
| `stockmem/scripts/evaluate_stockmem_feature_ablation.py` | Tắt từng feature block để xem ảnh hưởng. |
| `stockmem/scripts/experimental/evaluate_majority_consensus_retrievers.py` | Đánh giá retriever bằng `Majority@10`. |
| `stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py` | Chọn decision head trên validation. |
| `stockmem/scripts/experimental/train_majority_consensus_retriever.py` | Tune fusion weights cho learned/recency/fixed/regime. |
| `stockmem/scripts/retrain_finbert_retriever.py` | Fine-tune learned retriever; đã dùng cho ETH. |
| `aihub/scripts/evaluate_naive_llm_baseline.py` | Baseline LLM chỉ dùng context hiện tại, không dùng StockMem. |
| `stockmem/scripts/run_submission_reproduction.py` | Orchestrator chạy lại các thí nghiệm chính. |

Artifact kết quả chính:

| Artifact | Nội dung |
| --- | --- |
| `artifacts/current_context_ai_eval/summary.json` | Baseline LLM không dùng historical retrieval. |
| `artifacts/learned_strict_test_v3/summary.json` | Strict fixed/learned comparison. |
| `artifacts/majority_consensus_retriever_eval_20260703/summary.json` | BTC majority retrieval audit. |
| `artifacts/consensus_retriever_heads_20260703/summary.json` | BTC decision-head search. |
| `artifacts/eth_zero_shot_consensus_heads_20260704/summary.json` | ETH zero-shot consensus result. |
| `artifacts/eth_learned_recency_h30_consensus_heads_20260704/summary.json` | ETH maintained fine-tuned result. |

Tài liệu hỗ trợ:

| File | Vai trò |
| --- | --- |
| `docs/stockmem/data_flow.md` | Luồng dữ liệu chi tiết. |
| `docs/stockmem/experiment_metrics_catalog.md` | Catalog số liệu thí nghiệm. |
| `docs/stockmem/experiments.md` | Diễn giải kết quả chính. |
| `docs/stockmem/multi_asset_stockmem_report.md` | Báo cáo BTC/ETH. |
| `docs/stockmem/reproducibility.md` | Lệnh Docker để reproduce. |

## 10. Thí Nghiệm Và Kết Quả

### 10.1 Baseline LLM Không Dùng Truy Xuất Lịch Sử

Mục tiêu là kiểm tra nếu chỉ dùng context hiện tại thì kết quả như thế nào.
Baseline này nhận market candle hiện tại, biến động gần nhất, sentiment một
ngày và một số tiêu đề tin tức rút gọn. Baseline không nhận evidence lịch sử từ
StockMem.

Kết quả BTC test:

| Model | n | Overall DA | Active DA | Coverage | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_current_ai` | 305 | 0.2787 | 0.4031 | 0.6426 | 0.6164 | 0.3574 | 0.0262 |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6295 | 0.2492 | 0.1213 |
| `knn_returns` | 305 | 0.2918 | 0.4146 | 0.6721 | 0.5574 | 0.3279 | 0.1148 |

Diễn giải: baseline LLM không dùng truy xuất lịch sử có tỷ lệ phát SELL rất
thấp (`SELL rate = 0.0262`). Điều này cho thấy current-context prompting không
đủ để tạo quyết định downside ổn định.

### 10.2 Strict Fixed/Learned Comparison

Kết quả BTC test:

| Model | n | Overall DA | Active DA | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 |
| `fixed_retriever_learned_head` | 305 | 0.3508 | 0.4500 | 0.8525 | 0.8361 |
| `learned_retriever_fixed_head` | 305 | 0.3148 | 0.4182 | 0.7213 | 0.8459 |
| `learned_finbert_rolling_stable` | 305 | 0.3410 | 0.4393 | 0.7836 | 0.8459 |

Diễn giải: learned retriever cải thiện một số chỉ số retrieval như Hit@5,
nhưng không tự động cải thiện decision accuracy. Kết quả này dẫn đến việc tách
retriever evaluation khỏi decision-head evaluation.

### 10.3 Majority@10 Evidence Retrieval

Metric `Hit@5` tương đối dễ vì chỉ cần có một record cùng hướng trong top-k.
Do đó hệ thống dùng thêm metric chặt hơn:

```text
Majority@10 = ít nhất 5/10 evidence records có cùng D7 class với query
```

Kết quả BTC:

| Retriever | Val Majority@10 | Test Majority@10 | Full-history Majority@10 |
| --- | ---: | ---: | ---: |
| `fixed_only` | 0.4368 | 0.3639 | 0.3817 |
| `learned_only` | 0.5057 | 0.3541 | 0.3755 |
| `recency_only` | 0.6264 | 0.5180 | 0.5075 |
| `learned_recency_50_50` | 0.5920 | 0.5443 | 0.5106 |

Diễn giải: pure learned retrieval chưa đủ. Recency là tín hiệu mạnh do thị
trường có tính liên tục theo xu hướng. Pipeline cuối cùng chọn
`learned_recency_50_50` vì kết hợp được learned historical similarity với trend
awareness.

### 10.4 Consensus Decision Head Cho BTC

Kết quả:

| Split | n | Overall DA | Active DA | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 174 | 0.6379 | 0.7658 | 0.9080 | 0.7614 | 0.0345 | 0.7544 | 0.5920 |
| Test | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | 0.7114 | 0.5443 |

Điểm mạnh là SELL DA cao (`0.7114`). Hạn chế là HOLD DA thấp (`0.0000`), nên
mô hình hiện tại nên được mô tả là hệ thống directional decision có coverage
cao, không phải classifier ba lớp cân bằng.

### 10.5 ETH Zero-Shot

ETH được đánh giá trước bằng artifact BTC để kiểm tra khả năng chuyển giao.

ETH data:

| Field | Value |
| --- | ---: |
| Raw rows | 2908 |
| Rows có matured D7 | 2903 |
| Date range | `2018-01-05` đến `2026-07-01` |
| Validation rows | 174 |
| Test rows | 305 |

Kết quả ETH zero-shot:

| Profile | n | Overall DA | Active DA | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC artifact on ETH | 305 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |

Diễn giải: artifact BTC vẫn tạo được tín hiệu trên ETH. Điều này cho thấy cơ
chế StockMem có khả năng chuyển giao, nhưng kết quả chưa phải tối ưu cho ETH.

### 10.6 ETH Fine-Tuning

ETH fine-tuning tạo artifact riêng:

```text
stockmem/config/learned_retriever_finbert.eth.json
```

ETH maintained profile:

```text
retriever: eth_learned_recency_50_50_h30
head:      mean_learned_weights_buy0.50_sell0.75
```

Kết quả:

| Split | n | Overall DA | Active DA | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 174 | 0.6494 | 0.7530 | 0.9540 | 0.5472 | 0.0714 | 0.8817 | 0.5575 |
| Test | 305 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.0526 | 0.6496 | 0.5246 |

So với zero-shot:

```text
overall:     0.5344 -> 0.6000
active:      0.6014 -> 0.6793
BUY DA:      0.5769 -> 0.7077
SELL DA:     0.6204 -> 0.6496
Majority@10: 0.4754 -> 0.5246
```

Diễn giải: fine-tuning riêng cho ETH cải thiện pipeline cuối cùng. Điều này ủng
hộ thiết kế multi-asset theo profile riêng thay vì dùng một artifact chung cho
tất cả asset.

## 11. Những Kết Luận Có Thể Báo Cáo

Các kết luận hiện tại nên được trình bày ở mức vừa phải:

1. StockMem hiệu quả hơn baseline LLM chỉ dùng context hiện tại trên BTC test.
2. StockMem cung cấp evidence lịch sử có thể audit được cho từng prediction.
3. Pure learned retrieval chưa đủ; learned similarity cần kết hợp recency/trend
   awareness.
4. ETH zero-shot cho thấy cơ chế có khả năng chuyển giao.
5. ETH fine-tuning cải thiện rõ pipeline cuối cùng.
6. BTC và ETH nên dùng profile riêng trong cùng một kiến trúc chung.

## 12. Hạn Chế

Các hạn chế cần nêu rõ:

1. HOLD classification còn yếu. BTC test HOLD DA là `0.0000`, ETH test HOLD DA
   là `0.0526`.
2. Recency là tín hiệu mạnh, nên mô hình có rủi ro trong các giai đoạn đảo
   chiều nhanh.
3. Tập test hiện tại có 305 dòng, phù hợp cho đánh giá đồ án ứng dụng nhưng
   chưa đủ để khẳng định như một hệ thống giao dịch hoàn chỉnh.
4. Kết quả hiện tại tập trung vào directional accuracy và evidence quality,
   chưa tối ưu trực tiếp lợi nhuận giao dịch.
5. Một số thí nghiệm như hybrid reranking và head-aligned retriever là kết quả
   âm; chúng nên được dùng để chứng minh quá trình nghiên cứu, không dùng làm
   claim chính.

## 13. Điểm Mạnh Của Phần StockMem

Các điểm có thể nhấn mạnh khi trình bày:

1. Có luồng dữ liệu rõ ràng từ raw record đến prediction.
2. Mỗi prediction có thể truy vết top-10 evidence records.
3. Có kiểm soát rò rỉ dữ liệu tương lai bằng matured pool.
4. Có baseline không dùng retrieval để so sánh.
5. Có ablation và negative results, thể hiện quá trình thử nghiệm đầy đủ.
6. Có mở rộng multi-asset từ BTC sang ETH.
7. Có Docker/reproducibility docs để chạy lại thí nghiệm.

## 14. Câu Kết Luận Đề Xuất

Phiên bản đầy đủ:

```text
StockMem là một tầng historical memory có cấu trúc cho dự đoán hướng đi crypto.
Thay vì để mô hình suy luận trực tiếp từ dữ liệu hiện tại, hệ thống truy xuất
các trạng thái thị trường tương tự trong quá khứ, tổng hợp outcome D7 của các
trạng thái đó, rồi đưa ra tín hiệu BUY/HOLD/SELL bằng decision head được chọn
trên validation. Kết quả cho thấy StockMem vượt baseline LLM không dùng truy
xuất lịch sử trên BTC, đạt active DA khoảng 68% trên BTC/ETH, và có thể mở rộng
sang ETH bằng fine-tuning riêng cho từng asset.
```

Phiên bản ngắn cho slide:

```text
StockMem = structured market memory + learned/recency retrieval + evidence
consensus head. Điểm mạnh là dự đoán có bằng chứng lịch sử, deterministic,
audit được, và mở rộng được từ BTC sang ETH bằng asset-specific profile.
```

## 15. File Nên Mở Khi Cần Giải Thích Thêm

| Câu hỏi | File nên mở |
| --- | --- |
| Luồng dữ liệu đi như thế nào? | `docs/stockmem/data_flow.md` |
| Số liệu tổng hợp ở đâu? | `docs/stockmem/experiment_metrics_catalog.md` |
| Kết quả BTC/ETH chính? | `docs/stockmem/multi_asset_stockmem_report.md` |
| ETH fine-tune ra sao? | `docs/stockmem/eth_finetune.md` |
| Chạy lại thí nghiệm thế nào? | `docs/stockmem/reproducibility.md` |
| Paper học thuật dài hơn? | `docs/stockmem/academic_paper.md` |
