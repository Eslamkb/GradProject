import TableCsv from "./TableCsv.js";

const tableRoot = document.querySelector("#csvRoot");
const csvFileInput = document.querySelector("#file_input");
const tableCsv = new TableCsv(tableRoot);

csvFileInput.addEventListener("change", e => {

    Papa.parse(csvFileInput.files[0], {
        delimiter: ",",
        skipEmptyLines: true,
        complete: results => {
            tableCsv.update(results.data.slice(1, 501), results.data[0]);
        }
    });

});

