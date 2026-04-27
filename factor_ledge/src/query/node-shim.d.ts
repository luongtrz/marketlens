declare module "http" {
  export const createServer: any;
  export type IncomingMessage = any;
  export type ServerResponse = any;
}

declare module "fs" {
  export const existsSync: (path: string) => boolean;
  export const readFileSync: (path: string, encoding: string) => string;
}

declare module "path" {
  export const join: (...parts: string[]) => string;
}

declare const process: {
  cwd: () => string;
  env: Record<string, string | undefined>;
};
