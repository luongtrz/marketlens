import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { CryptoModule } from './crypto/crypto.module';
import { AiModule } from './ai/ai.module';
import { RealtimeModule } from './realtime/realtime.module';
import { MarketCandle } from './crypto/entities/market-candle.entity';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
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
    CryptoModule,
    AiModule,
    RealtimeModule,
  ],
})
export class AppModule { }
