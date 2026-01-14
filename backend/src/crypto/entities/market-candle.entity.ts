import { Entity, Column, PrimaryColumn, Index } from 'typeorm';

@Entity('market_candles')
@Index(['symbol', 'resolution', 'timestamp'], { unique: true })
export class MarketCandle {
    @PrimaryColumn({ type: 'varchar', length: 20 })
    symbol: string;

    @PrimaryColumn({ type: 'varchar', length: 10 })
    resolution: string; // '1m', '1h', '1d'

    @PrimaryColumn({ type: 'bigint' })
    timestamp: number; // Unix timestamp in milliseconds

    @Column({ type: 'decimal', precision: 20, scale: 8 })
    open: number;

    @Column({ type: 'decimal', precision: 20, scale: 8 })
    high: number;

    @Column({ type: 'decimal', precision: 20, scale: 8 })
    low: number;

    @Column({ type: 'decimal', precision: 20, scale: 8 })
    close: number;

    @Column({ type: 'decimal', precision: 30, scale: 8 })
    volume: number;
}
