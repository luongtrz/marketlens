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
                host: configService.get('DATABASE_HOST', 'db'),
                port: configService.get('DATABASE_PORT', 5432),
                username: configService.get('POSTGRES_USER', 'postgres'),
                password: configService.get('POSTGRES_PASSWORD', 'postgres'),
                database: configService.get('POSTGRES_DB', 'marketlens'),
                entities: [MarketCandle],
                synchronize: true, // Auto-create tables (disable in production)
            }),
        }),
    ],
})
export class DatabaseModule { }
