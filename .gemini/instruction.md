# MarketLens - Real-time Cryptocurrency Dashboard

## THONG TIN TONG QUAN

### Cau truc du an
- **Frontend**: React + TypeScript + Vite (localhost:5173)
- **Backend**: NestJS + TypeScript (localhost:3001)
- **Real-time**: Socket.IO + Binance WebSocket

### Muc tieu hien tai
Toi uu hoa cap nhat du lieu real-time giong Binance de hien thi gia cryptocurrency cap nhat lien tuc.

## CAU TRUC WEBSOCKET HIEN TAI

### Backend - RealtimeGateway
**File**: `/backend/src/realtime/realtime.gateway.ts`

Tinh nang da co:
- Ket noi voi Binance WebSocket: `wss://stream.binance.com:9443/ws`
- Ho tro 2 loai du lieu real-time:
  - **Trade**: Cap nhat giao dich tuc thoi
  - **Kline**: Cap nhat nen 1 phut (candlestick)
- Tu dong reconnect khi mat ket noi
- Room-based subscription (client chi nhan du lieu cua coin da subscribe)

### Frontend - Dashboard.tsx
**File**: `/frontend/pages/Dashboard.tsx`

Tinh nang da co (dong 162-286):
- Ket noi Socket.IO voi namespace 'realtime'
- Subscribe trade va kline updates cho coin dang chon
- Throttle trade updates (200ms) de tranh re-render qua nhieu
- Cap nhat chart realtime voi du lieu moi
- Tu dong cap nhat gia hien thi o header

### API Service
**File**: `/frontend/services/apiService.ts`

- `createSocketConnection()`: Tao ket noi WebSocket
- Su dung Socket.IO client voi transport: websocket
- Co auto-reconnection

## CACH HOAT DONG

1. User chon mot coin (BTC, ETH, etc.)
2. Frontend gui event `join-room` voi symbol va type (trade/kline)
3. Backend subscribe stream tuong ung tu Binance:
   - Trade: `btcusdt@trade`
   - Kline: `btcusdt@kline_1m`
4. Binance gui du lieu -> Backend parse -> Emit toi client room tuong ung
5. Frontend nhan du lieu va cap nhat:
   - Price hien tai
   - OHLC cua candlestick cuoi cung
   - High/Low neu co thay doi

## CAC VAN DE CO THE GAP

### 1. Performance Issues
- Qua nhieu updates gay lag
- **Giai phap**: Da co throttle 200ms cho trade updates

### 2. Memory Leaks
- Khong cleanup socket khi component unmount
- **Giai phap**: Da co cleanup trong useEffect return

### 3. Reconnection
- Mat ket noi khi internet mat
- **Giai phap**: Da co auto-reconnect sau 5s

## TOI UU HOA CO THE THUC HIEN

### 1. Buffer Updates
Cap nhat nhieu thay doi trong 1 batch thay vi tung ca nhan.

### 2. WebWorker
Xu ly du lieu WebSocket trong worker de tranh block main thread.

### 3. Selective Updates
Chi cap nhat data thay doi, khong re-render toan bo chart.

### 4. Compression
Su dung WebSocket compression de giam bandwidth.

## KIEM TRA HOAT DONG

Xem console log:
- "Connected to Realtime Socket"
- "Trade Update: ..."
- "WS Update: ..."

Neu khong thay log tren, kiem tra:
1. Backend co chay chua? (npm run start:dev)
2. WebSocket port co bi chan khong?
3. Environment variable VITE_API_URL co dung khong?

## BIEN MOI TRUONG

### Frontend (.env.local)
```
VITE_API_URL=http://localhost:3001/api
```

### Backend (.env)
Khong can cau hinh dac biet cho WebSocket, NestJS tu dong xu ly.

## DEBUG

Neu gap van de:
1. Mo Chrome DevTools -> Network -> WS -> Xem WebSocket frames
2. Kiem tra console log o ca Frontend va Backend
3. Test voi 1 coin don gian truoc (BTC)
4. Dam bao khong bi CORS issues

## GHI CHU QUAN TRONG

- WebSocket da hoat dong tot, du lieu dang cap nhat realtime
- Dashboard da co Live badge (xanh la) de hien thi trang thai
- Price cap nhat lien tuc theo du lieu tu Binance
- Chart candlestick cap nhat theo kline 1m interval
