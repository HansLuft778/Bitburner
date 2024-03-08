import { NS, NetscriptPort } from "@ns";

interface Port {
    portNumber: number;
    parentPID: number;
    identifier: string;
}

export class PortManager {
    private static instance: PortManager;

    private registeredPorts: Port[] = [];
    private lastPort = 0;

    public static getInstance(): PortManager {
        if (!PortManager.instance) {
            PortManager.instance = new PortManager();
        }
        return PortManager.instance;
    }

    public registerPort(ns: NS, port: number, pid: number, identifier = ""): Port {
        const portObj = {
            portNumber: port,
            parentPID: pid,
            identifier: identifier,
        };
        this.registeredPorts.push(portObj);

        return portObj;
    }

    public getPort(ns: NS, pid: number, identifier = ""): NetscriptPort {
        const portNum = this.getNextPort();

        const port = {
            portNumber: portNum,
            parentPID: pid,
            identifier: identifier,
        };
        this.registeredPorts.push(port);

        return ns.getPortHandle(port.portNumber);
    }

    public unregisterPort(ns: NS, port: number): void {
        const index = this.registeredPorts.findIndex((p) => p.portNumber === port);
        if (index === -1) {
            throw new Error(`Port ${port} is not registered`);
        }
        this.registeredPorts.splice(index, 1);

        ns.clearPort(port);
    }

    public reset(): void {
        this.registeredPorts = [];
        this.lastPort = 0;
    }

    private getNextPort(): number {
        this.lastPort++;
        return this.lastPort;
    }
}
