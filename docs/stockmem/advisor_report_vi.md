# Báo cáo StockMem: Ý tưởng, Cách Thực Hiện, Luồng Dữ Liệu Và Kết Quả

Tài liệu này dùng để trình bày với giảng viên hướng dẫn về phần StockMem trong
MarketLens. Mục tiêu là giải thích rõ ý tưởng, cách hệ thống hoạt động, dữ liệu
đi qua những bước nào, đã triển khai bằng những file nào, và kết quả thực
nghiệm hiện tại chứng minh được điều gì.

## 1. Tóm Tắt Ngắn Gọn

StockMem là một tầng trí nhớ lịch sử có cấu trúc cho bài toán dự đoán hướng đi
thị trường crypto. Thay vì chỉ đưa dữ liệu hiện tại vào AI rồi yêu cầu mô hình
ngôn ngữ tự suy luận, StockMem lưu lại các trạng thái thị trường trong quá khứ,
truy xuất những ngày lịch sử giống với ngày hiện tại, rồi dùng kết quả thực tế
sau đó của các ngày lịch sử này làm bằng chứng cho quyết định hiện tại.

Bài toán hiện tại là dự đoán nhãn D7:

```text
BUY  nếu future_return_7d > +2%
SELL nếu future_return_7d < -2%
HOLD nếu nằm trong khoảng [-2%, +2%]
```

Kết quả chính:

| Asset | Retriever chính | Decision head | Test rows | Overall DA | Active DA | Coverage | BUY DA | SELL DA | Majority@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | `learned_recency_50_50` | `count_vote_buy3_sell4` | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.7114 | 0.5443 |
| ETH | `eth_learned_recency_50_50_h30` | `mean_learned_weights_buy0.50_sell0.75` | 305 | 0.6000 | 0.6793 | 0.9508 | 0.7077 | 0.6496 | 0.5246 |

Kết luận ngắn: StockMem không chỉ là một mô hình dự đoán đơn lẻ. Nó là một
pipeline RAG có cấu trúc cho dữ liệu thị trường: lưu memory, truy xuất evidence,
tổng hợp evidence, rồi ra quyết định. Kết quả cho thấy pipeline này hiệu quả hơn
naive AI baseline trên BTC và có thể mở rộng sang ETH bằng fine-tuning riêng.

## 2. Vấn Đề Cần Giải Quyết

Với dữ liệu tài chính, nếu chỉ đưa ngữ cảnh hiện tại cho LLM, mô hình dễ gặp ba
vấn đề:

1. Không có trí nhớ lịch sử có kiểm soát.
2. Khó biết ngày hiện tại giống giai đoạn nào trong quá khứ.
3. Dễ thiên lệch theo ngôn ngữ tin tức hiện tại, đặc biệt là không nhận diện tốt
   chiều SELL.

StockMem giải quyết bằng cách biến mỗi ngày trong quá khứ thành một bản ghi
memory có cấu trúc:

```text
ngày, symbol, snapshot thị trường, vector sự kiện,
vector factor, vector indicator, vector giá,
future_return_1d/3d/7d/15d/30d
```

Khi cần dự đoán một ngày mới, hệ thống không hỏi AI một cách trống. Nó hỏi:

```text
Trong lịch sử, những ngày nào giống ngày hiện tại?
Sau các ngày đó 7 ngày, thị trường đã đi lên, đi xuống, hay đi ngang?
```

## 3. Ý Tưởng Chính

Ý tưởng của StockMem có thể hiểu như một hệ thống Retrieval-Augmented
Generation/Reasoning cho time series tài chính, nhưng thay vì truy xuất văn bản
thuần, hệ thống truy xuất các trạng thái thị trường có vector cấu trúc.

Luồng tư duy:

```text
current market state
  -> tìm các historical states tương tự
  -> lấy kết quả D7 thật của các historical states
  -> tạo một evidence set
  -> decision head tổng hợp evidence
  -> BUY/HOLD/SELL
```

Điểm quan trọng là hệ thống tách thành hai phần:

1. **Retriever**: chịu trách nhiệm tìm bằng chứng lịch sử phù hợp.
2. **Decision head**: chịu trách nhiệm biến bằng chứng thành tín hiệu.

Cách tách này giúp dễ kiểm thử. Nếu kết quả chưa tốt, ta biết vấn đề nằm ở
retrieval hay ở head ra quyết định.

## 4. Luồng Dữ Liệu Tổng Thể

Luồng dữ liệu đầy đủ:

```text
raw market/news/factor data
  -> StockMemRecord
  -> vector blocks
  -> historical candidate pool
  -> retriever scoring
  -> top-10 evidence records
  -> decision head
  -> prediction BUY/HOLD/SELL
  -> evaluation metrics
  -> report artifacts
```

Luồng triển khai hiện tại:

```text
BTC:
  data/exports/stockmem_records.ndjson
  -> stockmem/config/learned_retriever_finbert.json
  -> stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
  -> count_vote_buy3_sell4
  -> BTC prediction/evaluation

ETH:
  data/exports/stockmem_records_eth.ndjson
  -> stockmem/config/learned_retriever_finbert.eth.json
  -> stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
  -> mean_learned_weights_buy0.50_sell0.75
  -> ETH prediction/evaluation
```

File profile chung:

```text
stockmem/config/model_profiles.json
```

File này giúp endpoint biết symbol nào dùng artifact nào. BTC và ETH không bị
ép dùng chung một mô hình, nhưng vẫn dùng chung kiến trúc.

## 5. Các Loại Dữ Liệu Trong Một Memory Record

Một `StockMemRecord` có các nhóm thông tin chính:

| Thành phần | Ý nghĩa | Công dụng |
| --- | --- | --- |
| `date`, `symbol` | Ngày và asset. | Lọc đúng thị trường và đúng thời điểm. |
| `market_snapshot` | Trạng thái thị trường tại ngày đó. | Nguồn cho indicator, price, regime. |
| `event_vec` | Vector hóa sự kiện/tin tức. | Hỗ trợ learned retriever hiểu ngữ cảnh sự kiện. |
| `factor_vec` | Vector factor và taxonomy sự kiện. | So sánh ngữ cảnh thị trường/factor. |
| `indicator_vec` | Chỉ báo kỹ thuật/sentiment ngắn gọn. | So sánh trạng thái kỹ thuật. |
| `price_vec` | Biến động giá, volume, range. | So sánh market-state và xu hướng. |
| `future_return_7d` | Return thật sau 7 ngày. | Tạo nhãn BUY/HOLD/SELL khi đánh giá và khi dùng record lịch sử làm evidence. |

File định nghĩa record:

```text
stockmem/src/models.py
```

File tạo vector:

```text
stockmem/src/search/embedder.py
stockmem/src/search/event_memory.py
stockmem/src/search/taxonomy.py
```

## 6. Cách Truy Xuất Bằng Chứng

### 6.1 Fixed kNN

Phiên bản ban đầu dùng fixed kNN. Mỗi query và candidate được so sánh bằng
weighted cosine similarity trên ba block:

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
| `stockmem/scripts/optimize_weights.py` | Script tune weights. |

Fixed kNN có ưu điểm là ổn định và deterministic: cùng input, cùng weights, cùng
candidate pool thì kết quả giống nhau.

### 6.2 Learned Retriever

Learned retriever học một metric thay vì dùng weights thủ công. Công thức tổng
quát:

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

Sau khi artifact được tạo, learned retriever cũng deterministic. Truy vấn 100
lần với cùng dữ liệu và cùng artifact sẽ ra cùng ranking.

### 6.3 Learned + Recency Retriever

Kết quả thực nghiệm cho thấy pure learned retrieval chưa đủ tốt. Lý do là thị
trường có tính liên tục theo thời gian: các ngày gần hiện tại thường có thông
tin regime/trend quan trọng. Vì vậy pipeline hiện tại dùng learned retriever
kết hợp recency:

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

Các file config:

```text
stockmem/config/majority_consensus_retriever.learned_recency_50_50.json
stockmem/config/majority_consensus_retriever.eth.learned_recency_50_50_h30.json
```

Ý nghĩa: mô hình vừa học sự tương đồng lịch sử, vừa aware trend gần đây. Đây là
điểm cân bằng giữa “chỉ nhìn quá khứ xa” và “chỉ chạy theo gần đây”.

## 7. Decision Head

Retriever chỉ trả về evidence. Decision head mới là phần biến evidence thành
tín hiệu.

### BTC head

BTC dùng:

```text
count_vote_buy3_sell4
```

Luật:

```text
SELL nếu sell_count >= 4 và sell_count >= buy_count
BUY  nếu buy_count  >= 3 và buy_count  >  sell_count
HOLD nếu không thỏa hai điều kiện trên
```

Input là top-10 evidence từ `learned_recency_50_50`.

### ETH head

ETH dùng:

```text
mean_learned_weights_buy0.50_sell0.75
```

Input là top-10 evidence từ `eth_learned_recency_50_50_h30`. Head này được chọn
trên validation sau khi fine-tune retriever riêng cho ETH.

File chọn/tune head:

```text
stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py
```

## 8. Cách Kiểm Soát Leakage

Đây là phần rất quan trọng để bảo vệ tính đúng đắn của thí nghiệm.

Khi dự đoán ngày `t`, hệ thống chỉ được dùng record lịch sử mà kết quả D7 đã
biết tại thời điểm `t`.

Quy tắc:

```text
candidate.date < query.date
candidate.date + 7 days <= query.date
```

Nếu candidate quá gần query, future_return_7d của candidate chưa “mature”, nên
không được dùng làm evidence.

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

Test set có 305 rows. Validation set có 174 rows.

## 9. Các Script Chính

| Script | Chức năng |
| --- | --- |
| `stockmem/scripts/evaluate_stockmem_strict_models.py` | So sánh strict fixed/learned retriever và head. |
| `stockmem/scripts/evaluate_stockmem_feature_ablation.py` | Tắt từng feature block để xem ảnh hưởng. |
| `stockmem/scripts/experimental/evaluate_majority_consensus_retrievers.py` | Đánh giá retriever bằng `Majority@10`. |
| `stockmem/scripts/experimental/evaluate_consensus_retriever_heads.py` | Chọn decision head trên validation. |
| `stockmem/scripts/experimental/train_majority_consensus_retriever.py` | Tune fusion weights cho learned/recency/fixed/regime. |
| `stockmem/scripts/retrain_finbert_retriever.py` | Fine-tune learned retriever, đã dùng cho ETH. |
| `aihub/scripts/evaluate_naive_llm_baseline.py` | Baseline AI ngây thơ chỉ dùng context hiện tại, không dùng StockMem. |
| `stockmem/scripts/run_submission_reproduction.py` | Orchestrator chạy lại các thí nghiệm chính. |

## 10. Các Artifact Và Báo Cáo Chính

| File | Vai trò |
| --- | --- |
| `docs/stockmem/data_flow.md` | Mô tả chi tiết luồng dữ liệu và file liên quan. |
| `docs/stockmem/experiment_metrics_catalog.md` | Catalog số liệu các thí nghiệm đã chạy. |
| `docs/stockmem/experiments.md` | Tóm tắt kết quả và diễn giải thí nghiệm. |
| `docs/stockmem/multi_asset_stockmem_report.md` | Báo cáo BTC/ETH multi-asset. |
| `docs/stockmem/eth_zero_shot.md` | Báo cáo ETH zero-shot. |
| `docs/stockmem/eth_finetune.md` | Báo cáo ETH fine-tune. |
| `docs/stockmem/reproducibility.md` | Lệnh Docker để reproduce. |

Artifact kết quả chính:

| Artifact | Nội dung |
| --- | --- |
| `artifacts/current_context_ai_eval/summary.json` | Naive LLM baseline. |
| `artifacts/learned_strict_test_v3/summary.json` | Strict fixed/learned comparison. |
| `artifacts/majority_consensus_retriever_eval_20260703/summary.json` | BTC majority retrieval audit. |
| `artifacts/consensus_retriever_heads_20260703/summary.json` | BTC decision-head search. |
| `artifacts/eth_zero_shot_consensus_heads_20260704/summary.json` | ETH zero-shot consensus result. |
| `artifacts/eth_learned_recency_h30_consensus_heads_20260704/summary.json` | ETH maintained fine-tuned result. |

## 11. Thí Nghiệm 1: Naive AI Baseline

Mục tiêu: kiểm tra nếu chỉ đưa context hiện tại cho AI thì kết quả như thế nào.

Naive AI nhận:

```text
current market candle,
yesterday change percent,
1-day news sentiment,
một số raw news title rút gọn
```

Naive AI không nhận historical retrieval evidence.

Kết quả BTC test:

| Model | n | Overall DA | Active DA | Coverage | BUY rate | HOLD rate | SELL rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_current_ai` | 305 | 0.2787 | 0.4031 | 0.6426 | 0.6164 | 0.3574 | 0.0262 |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.6295 | 0.2492 | 0.1213 |
| `knn_returns` | 305 | 0.2918 | 0.4146 | 0.6721 | 0.5574 | 0.3279 | 0.1148 |

Ý nghĩa: naive AI rất ít phát SELL (`SELL rate = 0.0262`). Điều này cho thấy
AI chỉ nhìn context hiện tại có xu hướng tránh dự đoán downside, trong khi
StockMem có thể dùng lịch sử để nhận diện SELL tốt hơn.

## 12. Thí Nghiệm 2: Strict Fixed/Learned Comparison

Mục tiêu: so sánh fixed kNN, learned retriever, learned head trong setup strict
ban đầu.

Kết quả BTC test:

| Model | n | Overall DA | Active DA | Coverage | Hit@5 same sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_knn_rolling_stable` | 305 | 0.3180 | 0.4236 | 0.7508 | 0.8361 |
| `fixed_retriever_learned_head` | 305 | 0.3508 | 0.4500 | 0.8525 | 0.8361 |
| `learned_retriever_fixed_head` | 305 | 0.3148 | 0.4182 | 0.7213 | 0.8459 |
| `learned_finbert_rolling_stable` | 305 | 0.3410 | 0.4393 | 0.7836 | 0.8459 |

Diễn giải: learned retriever cải thiện một số metric retrieval như Hit@5,
nhưng không tự động làm decision tốt hơn. Learned head kết hợp fixed retriever
là hướng tốt hơn trong strict setup cũ. Sau đó, pipeline chuyển sang đánh giá
bằng evidence-consensus vì Hit@5 quá dễ.

## 13. Thí Nghiệm 3: Majority@10 Evidence Retrieval

Vấn đề của Hit@5 là chỉ cần top-5 có một record cùng hướng là đã được tính tốt.
Metric này quá dễ. Vì vậy dùng metric chặt hơn:

```text
Majority@10 = ít nhất 5/10 evidence record có cùng D7 class với query
```

Kết quả BTC:

| Retriever | Val Majority@10 | Test Majority@10 | Full-history Majority@10 |
| --- | ---: | ---: | ---: |
| `fixed_only` | 0.4368 | 0.3639 | 0.3817 |
| `learned_only` | 0.5057 | 0.3541 | 0.3755 |
| `recency_only` | 0.6264 | 0.5180 | 0.5075 |
| `learned_recency_50_50` | 0.5920 | 0.5443 | 0.5106 |

Ý nghĩa: pure learned không đủ. Pure recency mạnh vì thị trường có trend
continuity. Nhưng model cuối cùng chọn `learned_recency_50_50` vì nó vừa giữ
memory lịch sử vừa có trend awareness và đạt test/full-history tốt.

## 14. Thí Nghiệm 4: Consensus Decision Head

Sau khi chọn retriever, hệ thống chọn head để tổng hợp top-10 evidence.

BTC head tốt nhất:

```text
count_vote_buy3_sell4
```

Kết quả:

| Split | n | Overall DA | Active DA | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 174 | 0.6379 | 0.7658 | 0.9080 | 0.7614 | 0.0345 | 0.7544 | 0.5920 |
| Test | 305 | 0.5475 | 0.6826 | 0.9607 | 0.6224 | 0.0000 | 0.7114 | 0.5443 |

Điểm mạnh: SELL DA rất tốt (`0.7114`). Điểm yếu: HOLD DA bằng `0.0000`, nghĩa
là mô hình hiện tại nên được mô tả là directional decision system, không phải
balanced three-class classifier.

## 15. Thí Nghiệm 5: ETH Zero-Shot

Mục tiêu: kiểm tra artifact BTC có chuyển sang ETH được không trước khi train
riêng.

ETH data:

| Field | Value |
| --- | ---: |
| Raw rows | 2908 |
| Rows có matured D7 | 2903 |
| Date range | `2018-01-05` đến `2026-07-01` |
| Validation rows | 174 |
| Test rows | 305 |

ETH zero-shot với BTC learned-recency pipeline:

| Profile | n | Overall DA | Active DA | Coverage | BUY DA | HOLD DA | SELL DA | Majority@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC artifact on ETH | 305 | 0.5344 | 0.6014 | 0.9705 | 0.5769 | 0.0789 | 0.6204 | 0.4754 |

Ý nghĩa: ngay cả khi chưa fine-tune ETH, StockMem vẫn chuyển được một phần sang
ETH. Điều này ủng hộ hướng multi-asset, nhưng kết quả chưa tối ưu nên cần
fine-tune riêng.

## 16. Thí Nghiệm 6: ETH Fine-Tuning

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

Ý nghĩa: fine-tuning riêng cho ETH có tác dụng rõ trên pipeline cuối cùng. Đây
là bằng chứng rằng kiến trúc StockMem có thể mở rộng multi-asset nhưng nên có
profile riêng cho từng asset.

## 17. Những Gì Kết Quả Hiện Tại Chứng Minh Được

Các kết luận có thể trình bày tương đối chắc:

1. **Structured StockMem tốt hơn naive AI prompting** trong bối cảnh BTC test.
   Naive AI thiếu historical evidence và gần như không phát SELL.
2. **StockMem hoạt động như một RAG có cấu trúc**, không chỉ là một classifier.
   Evidence top-10 có thể audit được.
3. **Pure learned retrieval chưa đủ**, nhưng learned retrieval kết hợp recency
   là hướng tốt hơn.
4. **Recency là tín hiệu quan trọng**, nhưng không nên nói mô hình chỉ dùng
   recency. Pipeline cuối vẫn giữ learned similarity để bám vào memory lịch sử.
5. **ETH zero-shot có tín hiệu tốt**, chứng minh cơ chế không chỉ dành cho BTC.
6. **ETH fine-tuning cải thiện pipeline cuối**, chứng minh cần asset-specific
   profile.

## 18. Hạn Chế Hiện Tại

Các hạn chế nên nói thẳng trong báo cáo:

1. HOLD classification còn yếu. BTC maintained head có HOLD DA `0.0000`; ETH
   cũng chỉ `0.0526`.
2. Recency là tín hiệu mạnh, nên mô hình có rủi ro khi thị trường đảo chiều.
3. Dataset test hiện tại là 305 rows, đủ cho đồ án ứng dụng nhưng chưa phải quy
   mô rất lớn.
4. Kết quả hiện tại tập trung vào directional accuracy/evidence quality, chưa
   phải tối ưu PnL giao dịch.
5. Một số thí nghiệm cũ như hybrid reranking và head-aligned retriever là kết
   quả âm, nên chỉ nên dùng để chứng minh quá trình nghiên cứu, không dùng làm
   claim chính.

## 19. Điểm Mạnh Khi Bảo Vệ Đồ Án

Các điểm nên nhấn mạnh:

1. Pipeline có khả năng audit: mỗi prediction có thể xem top-10 evidence.
2. Có leakage control bằng matured pool.
3. Có baseline naive AI để chứng minh không phải chỉ cần prompt LLM là đủ.
4. Có ablation và negative result, cho thấy quá trình nghiên cứu trung thực.
5. Có multi-asset extension từ BTC sang ETH.
6. Có profile router để triển khai nhiều asset trong một kiến trúc chung.
7. Có Docker/reproducibility docs để chạy lại thí nghiệm.

## 20. Kết Luận Đề Xuất Khi Trình Bày

Câu kết luận chính:

```text
StockMem là một tầng historical memory có cấu trúc cho dự đoán hướng đi crypto.
Thay vì để AI suy luận trực tiếp từ dữ liệu hiện tại, hệ thống truy xuất các
trạng thái thị trường tương tự trong quá khứ, tổng hợp kết quả D7 của các trạng
thái đó, rồi ra tín hiệu BUY/HOLD/SELL bằng decision head đã chọn trên
validation. Kết quả cho thấy StockMem vượt naive current-context AI trên BTC,
đạt active DA khoảng 68% trên BTC/ETH, và có thể mở rộng multi-asset bằng
fine-tuning riêng cho từng asset.
```

Mô tả ngắn hơn nếu cần nói trong slide:

```text
StockMem = structured market memory + learned/recency retrieval + evidence
consensus head. Điểm mạnh là prediction có bằng chứng lịch sử, deterministic,
audit được, và có thể mở rộng từ BTC sang ETH.
```

## 21. File Nên Mở Khi Thầy Hỏi

| Câu hỏi | File nên mở |
| --- | --- |
| Luồng dữ liệu đi như thế nào? | `docs/stockmem/data_flow.md` |
| Số liệu tổng hợp ở đâu? | `docs/stockmem/experiment_metrics_catalog.md` |
| Kết quả BTC/ETH chính? | `docs/stockmem/multi_asset_stockmem_report.md` |
| ETH fine-tune ra sao? | `docs/stockmem/eth_finetune.md` |
| Chạy lại thí nghiệm thế nào? | `docs/stockmem/reproducibility.md` |
| Paper học thuật dài hơn? | `docs/stockmem/academic_paper.md` |

