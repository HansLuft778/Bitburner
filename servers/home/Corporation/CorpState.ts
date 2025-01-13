export class CorporationState {
    private constructor() {
        this.smartSupplyData = new Map<string, number>();
    }

    private static instance: CorporationState = null;

    private smartSupplyData: Map<string, number>;

    private setOfDivisionsWaitingForRP: Set<string> = new Set<string>();

    public static getInstance(): CorporationState {
        if (!CorporationState.instance)
            CorporationState.instance = new CorporationState();
        return CorporationState.instance;
    }

    public resetState() {
        this.smartSupplyData = new Map<string, number>();
        this.setOfDivisionsWaitingForRP = new Set<string>();
    }

    public getSmartSupplyData() {
        return this.smartSupplyData;
    }

    public setSmartSupplyData(key, value) {
        this.smartSupplyData.set(key, value);
    }

    public addDivisionToSet(division: string) {
        this.setOfDivisionsWaitingForRP.add(division);
    }
    public removeDivisionFromSet(division: string) {
        this.setOfDivisionsWaitingForRP.delete(division);
    }

    public getDivisionWaitingForRpSet(): Set<String> {
        return this.setOfDivisionsWaitingForRP;
    }
}
