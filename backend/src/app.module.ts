import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { CryptoModule } from './crypto/crypto.module';
import { AiModule } from './ai/ai.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    CryptoModule,
    AiModule,
  ],
})
export class AppModule { }
