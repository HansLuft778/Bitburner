class HttpClient {
    private baseURL: string = "";

    constructor(baseURL: string) {
        this.baseURL = baseURL;
    }

    async get(endpoint) {
        // ...existing code...
        const response = await fetch(`${this.baseURL}${endpoint}`);
        return response.json();
    }

    async post(endpoint, data) {
        // ...existing code...
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        return response.json();
    }

    async put(endpoint, data) {
        // ...existing code...
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        return response.json();
    }

    async delete(endpoint) {
        // ...existing code...
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: "DELETE"
        });
        return response.status;
    }
}

export default HttpClient;

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();
    ns.disableLog("ALL");

    const http = new HttpClient("http://127.0.0.1:8080");
}
