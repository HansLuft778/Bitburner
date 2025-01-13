async function fetchAsync(url: string) {
    const response = await fetch("http://127.0.0.1:8080");
    return await response.text();
}

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();
    ns.disableLog("ALL");
    ns.print("\n");

    ns.print(await fetchAsync("127.0.0.1"));
}
