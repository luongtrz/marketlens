import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { CryptoService } from './crypto.service';
import { CryptoController } from './crypto.controller';
import { ConfigModule } from '@nestjs/config';
import { MarketCandle } from './entities/market-candle.entity';

@Module({
  imports: [ConfigModule, TypeOrmModule.forFeature([MarketCandle])],
  controllers: [CryptoController],
  providers: [CryptoService],
})
export class CryptoModule { }
