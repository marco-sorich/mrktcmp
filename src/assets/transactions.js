/* ---------------------------------------------------------------------------
 * transactions.js – clientside callbacks for the transaction tables
 *
 * Scrolling a DOM element is something only the browser can do, so these run
 * clientside (referenced from src/callbacks/backtesting.py via
 * ClientsideFunction(namespace='transactions', ...)).
 *
 *   scrollButtons   – the "⤒ Top" / "⤓ Bottom" buttons jump the scrollable
 *                     table wrapper to its start or end.
 *   graphClickToRow – clicking a chart point activates the matching run's tab
 *                     and scrolls its table to (and highlights) the nearest
 *                     event row.
 * ------------------------------------------------------------------------- */

window.dash_clientside = window.dash_clientside || {};

(function () {
    "use strict";

    var NO_UPDATE = window.dash_clientside.no_update;

    /* Parse the JSON component id out of a triggered prop_id like
       '{"run":"a","type":"bt-tx-top"}.n_clicks'. Returns null on failure. */
    function parseTriggeredId(propId) {
        if (!propId) { return null; }
        var idStr = propId.substring(0, propId.lastIndexOf("."));
        try {
            return JSON.parse(idStr);
        } catch (e) {
            return null;
        }
    }

    /* Remove the highlight class from any previously highlighted row. */
    function clearHighlight() {
        var prev = document.querySelectorAll(".bt-tx-row.tx-row-highlight");
        prev.forEach(function (el) { el.classList.remove("tx-row-highlight"); });
    }

    /* Scroll the given run's table to a row index and highlight it. The tab was
       just activated, so we defer the lookup until React has re-rendered. */
    function scrollToRow(runId, rowIndex) {
        window.setTimeout(function () {
            var wrapper = document.getElementById("bt-tx-" + runId);
            if (!wrapper) { return; }
            var row = wrapper.querySelector('tr[data-index="' + rowIndex + '"]');
            if (!row) { return; }
            clearHighlight();
            row.classList.add("tx-row-highlight");
            row.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }, 90);
    }

    /* Find the index of the event row whose date is closest to a clicked x. */
    function nearestRowIndex(rows, clickedX) {
        var target = new Date(clickedX).getTime();
        var bestIdx = 0;
        var bestDiff = Infinity;
        for (var i = 0; i < rows.length; i++) {
            var diff = Math.abs(new Date(rows[i].date).getTime() - target);
            if (diff < bestDiff) {  // strict < keeps the FIRST row on ties
                bestDiff = diff;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    window.dash_clientside.transactions = {

        scrollButtons: function (topClicks, bottomClicks) {
            var ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {
                return NO_UPDATE;
            }
            var trig = ctx.triggered[0];
            if (trig.value == null) { return NO_UPDATE; }
            var id = parseTriggeredId(trig.prop_id);
            if (!id) { return NO_UPDATE; }
            var wrapper = document.getElementById("bt-tx-" + id.run);
            if (!wrapper) { return NO_UPDATE; }
            wrapper.scrollTop = (id.type === "bt-tx-top") ? 0 : wrapper.scrollHeight;
            return NO_UPDATE;
        },

        graphClickToRow: function (clickData, store) {
            if (!clickData || !clickData.points || !store || !store.order) {
                return NO_UPDATE;
            }
            var pt = clickData.points[0];
            var curve = pt.curveNumber;
            // Ignore clicks on the transient highlight marker (last trace).
            if (curve == null || curve >= store.order.length) {
                return NO_UPDATE;
            }
            var runId = store.order[curve];
            var rows = (store.rows && store.rows[runId]) || [];
            if (rows.length === 0) {
                // Still switch to the run's tab even if it has no rows.
                return "tx-" + runId;
            }
            scrollToRow(runId, nearestRowIndex(rows, pt.x));
            return "tx-" + runId;  // activate this run's tab
        }
    };
})();
