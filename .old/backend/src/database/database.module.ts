import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MarketCandle } from '../crypto/entities/market-candle.entity';

@Module({
    imports: [
        TypeOrmModule.forRootAsync({
            imports: [ConfigModule],
            inject: [ConfigService],
            useFactory: (configService: ConfigService) => ({
                type: 'postgres',
                host: configService.get('DB_HOST', 'localhost'),
                port: configService.get('DB_PORT', 5433),
                username: configService.get('DB_USER', 'postgres'),
                password: configService.get('DB_PASSWORD', 'postgres'),
                database: configService.get('DB_NAME', 'marketlens'),
                entities: [MarketCandle],
                synchronize: true, // Auto-create tables (disable in production)
            }),
        }),
    ],
})
export class DatabaseModule { }
