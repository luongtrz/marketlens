import { OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect } from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
export declare class RealtimeGateway implements OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect {
    server: Server;
    private logger;
    private binanceWs;
    private activeSubscriptions;
    private binanceStreamUrl;
    afterInit(server: Server): void;
    handleConnection(client: Socket): void;
    handleDisconnect(client: Socket): void;
    private connectToBinance;
    handleJoinRoom(client: Socket, payload: {
        symbol: string;
        type: 'trade' | 'kline';
    }): void;
    handleLeaveRoom(client: Socket, payload: {
        symbol: string;
        type: 'trade' | 'kline';
    }): void;
    private updateBinanceSubscriptions;
}
