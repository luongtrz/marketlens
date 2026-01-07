import {
    WebSocketGateway,
    WebSocketServer,
    OnGatewayInit,
    OnGatewayConnection,
    OnGatewayDisconnect,
    SubscribeMessage,
} from '@nestjs/websockets';
import { Logger } from '@nestjs/common';
import { Server, Socket } from 'socket.io';
import WebSocket from 'ws';

@WebSocketGateway({
    cors: {
        origin: '*',
    },
    namespace: 'realtime',
})
export class RealtimeGateway
    implements OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect {
    @WebSocketServer() server: Server;
    private logger: Logger = new Logger('RealtimeGateway');
    private binanceWs: WebSocket;
    private activeSubscriptions: Set<string> = new Set();
    private binanceStreamUrl = 'wss://stream.binance.com:9443/ws';

    afterInit(server: Server) {
        this.logger.log('Realtime Gateway Initialized');
        this.connectToBinance();
    }

    handleConnection(client: Socket) {
        this.logger.log(`Client connected: ${client.id}`);
    }

    handleDisconnect(client: Socket) {
        this.logger.log(`Client disconnected: ${client.id}`);
    }

    private connectToBinance() {
        this.binanceWs = new WebSocket(this.binanceStreamUrl);

        this.binanceWs.on('open', () => {
            this.logger.log('Connected to Binance WebSocket');
            // Resubscribe if connection was lost
            if (this.activeSubscriptions.size > 0) {
                this.updateBinanceSubscriptions();
            }
        });

        this.binanceWs.on('message', (data: WebSocket.Data) => {
            try {
                const message = JSON.parse(data.toString());
                // Broadcast to specific rooms based on symbol
                if (message.e === 'trade') {
                    this.server.to(`trade:${message.s}`).emit('trade', message);
                } else if (message.e === 'kline') {
                    const symbol = message.s;
                    // Standardize kline data for frontend
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
            } catch (e) {
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

    @SubscribeMessage('join-room')
    handleJoinRoom(client: Socket, payload: { symbol: string; type: 'trade' | 'kline' }) {
        const room = `${payload.type}:${payload.symbol.toUpperCase()}`;
        client.join(room);
        this.logger.log(`Client ${client.id} joined ${room}`);

        // Subscribe to Binance if not already
        const binanceStreamName = `${payload.symbol.toLowerCase()}@${payload.type === 'kline' ? 'kline_1m' : 'trade'}`;
        if (!this.activeSubscriptions.has(binanceStreamName)) {
            this.activeSubscriptions.add(binanceStreamName);
            this.updateBinanceSubscriptions();
        }
    }

    @SubscribeMessage('leave-room')
    handleLeaveRoom(client: Socket, payload: { symbol: string; type: 'trade' | 'kline' }) {
        const room = `${payload.type}:${payload.symbol.toUpperCase()}`;
        client.leave(room);
        this.logger.log(`Client ${client.id} left ${room}`);
    }

    private updateBinanceSubscriptions() {
        if (this.binanceWs.readyState === WebSocket.OPEN && this.activeSubscriptions.size > 0) {
            const payload = {
                method: 'SUBSCRIBE',
                params: Array.from(this.activeSubscriptions),
                id: 1,
            };
            this.binanceWs.send(JSON.stringify(payload));
        }
    }
}
