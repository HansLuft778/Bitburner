export class CorporationState {
    private constructor() {
        this.smartSupplyData = new Map<string, number>();
    }

    private static instance: CorporationState = null;

    private smartSupplyData: Map<string, number>;

    public static getInstance(): CorporationState {
        if (!CorporationState.instance)
            CorporationState.instance = new CorporationState();
        return CorporationState.instance;
    }

    public getSmartSupplyData() {
        return this.smartSupplyData;
    }

    public setSmartSupplyData(key, value) {
        this.smartSupplyData.set(key, value)
    }
}
