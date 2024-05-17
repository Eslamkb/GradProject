export default class {
    /**
     * 
     * @param {HTMLTableElement} root the table element which will display the CSV data.
     */
    constructor(root) {
        this.root = root;
    }

    /**
     * 
     * clears existing data in the table and replaces it with new data
     * 
     * @param {string[][]} data a 2d array of data to be used as the table body 
     * @param {string[]} headerColumns list of headings to be used 
     */
    update(data, headerColumns = []) {
        this.clear();
        this.setHeader(headerColumns);
        this.setBody(data);
    }

    /**
     * Clears all contents of the table (incl. the heaer)
     */
    clear() {
        this.root.innerHTML = "";
    }

    /**
     * Sets the table header
     * 
     * @param {string[]} headerColumns List of headings to be used
     */
    setHeader(headerColumns) {
        this.root.insertAdjacentHTML("afterbegin", `
            <thead>
                <tr>
                    ${headerColumns.map(text => `<th>${text}</th>`).join("")}
                </tr>
            </thead>
        `);
    }

    /**
     * 
     * Sets the table body
     * 
     * @param {string[][]} data A 2d array of data to be used as the table body 
     */
    setBody(data) {
        const rowsHtml = data.map(row => {
            return `
                <tr>
                    ${row.map(text => `<td>${text}</td>`).join("")}
                </tr
            `;
        });

        this.root.insertAdjacentHTML("beforeend", `
            <tbody>
                ${rowsHtml.join("")}
            </tbody>
        `);
    }
}