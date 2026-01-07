"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RealtimeGateway = void 0;
const websockets_1 = require("@nestjs/websockets");
const common_1 = require("@nestjs/common");
const socket_io_1 = require("socket.io");
const ws_1 = __importDefault(require("ws"));
let RealtimeGateway = class RealtimeGateway {
    server;
    logger = new common_1.Logger('RealtimeGateway');
    binanceWs;
    activeSubscriptions = new Set();
    binanceStreamUrl = 'wss://stream.binance.com:9443/ws';
    afterInit(server) {
        this.logger.log('Realtime Gateway Initialized');
        this.connectToBinance();
    }
    handleConnection(client) {
        this.logger.log(`Client connected: ${client.id}`);
    }
    handleDisconnect(client) {
        this.logger.log(`Client disconnected: ${client.id}`);
    }
    connectToBinance() {
        this.binanceWs = new ws_1.default(this.binanceStreamUrl);
        this.binanceWs.on('open', () => {
            this.logger.log('Connected to Binance WebSocket');
            if (this.activeSubscriptions.size > 0) {
                this.updateBinanceSubscriptions();
            }
        });
        this.binanceWs.on('message', (data) => {
            try {
                const message = JSON.parse(data.toString());
                if (message.e === 'trade') {
                    this.server.to(`trade:${message.s}`).emit('trade', message);
                }
                else if (message.e === 'kline') {
                    const symbol = message.s;
                    const kline = {
                        time: message.k.t,
                        open: parseFloat(message.k.o),
                        high: parseFloat(message.k.h),
                        low: parseFloat(message.k.l),
                        close: parseFloat(message.k.c),
                        volume: parseFloat(message.k.v),
                        isFinal: message.k.x
                    };
                    this.server.to(`kline:${symbol}`).emit('kline', { symbol, data: kline });
                }
            }
            catch (e) {
                this.logger.error('Error parsing Binance message', e);
            }
        });
        this.binanceWs.on('error', (error) => {
            this.logger.error('Binance WebSocket Error', error);
        });
        this.binanceWs.on('close', () => {
            this.logger.warn('Binance WebSocket Closed. Reconnecting in 5s...');
            setTimeout(() => this.connectToBinance(), 5000);
        });
    }
    handleJoinRoom(client, payload) {
        const room = `${payload.type}:${payload.symbol.toUpperCase()}`;
        client.join(room);
        this.logger.log(`Client ${client.id} joined ${room}`);
        const binanceStreamName = `${payload.symbol.toLowerCase()}@${payload.type === 'kline' ? 'kline_1m' : 'trade'}`;
        if (!this.activeSubscriptions.has(binanceStreamName)) {
            this.activeSubscriptions.add(binanceStreamName);
            this.updateBinanceSubscriptions();
        }
    }
    handleLeaveRoom(client, payload) {
        const room = `${payload.type}:${payload.symbol.toUpperCase()}`;
        client.leave(room);
        this.logger.log(`Client ${client.id} left ${room}`);
    }
    updateBinanceSubscriptions() {
        if (this.binanceWs.readyState === ws_1.default.OPEN && this.activeSubscriptions.size > 0) {
            const payload = {
                method: 'SUBSCRIBE',
                params: Array.from(this.activeSubscriptions),
                id: 1,
            };
            this.binanceWs.send(JSON.stringify(payload));
        }
    }
};
exports.RealtimeGateway = RealtimeGateway;
__decorate([
    (0, websockets_1.WebSocketServer)(),
    __metadata("design:type", socket_io_1.Server)
], RealtimeGateway.prototype, "server", void 0);
__decorate([
    (0, websockets_1.SubscribeMessage)('join-room'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [socket_io_1.Socket, Object]),
    __metadata("design:returntype", void 0)
], RealtimeGateway.prototype, "handleJoinRoom", null);
__decorate([
    (0, websockets_1.SubscribeMessage)('leave-room'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [socket_io_1.Socket, Object]),
    __metadata("design:returntype", void 0)
], RealtimeGateway.prototype, "handleLeaveRoom", null);
exports.RealtimeGateway = RealtimeGateway = __decorate([
    (0, websockets_1.WebSocketGateway)({
        cors: {
            origin: '*',
        },
        namespace: 'realtime',
    })
], RealtimeGateway);
//# sourceMappingURL=realtime.gateway.js.map