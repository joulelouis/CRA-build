// Initialize column selector functionality for sensitivity results
$(document).ready(function() {
    // Update hazard group checkbox handler
    $('.hazard-group').on('change', function() {
        const isChecked = $(this).prop('checked');
        const hazardClass = $(this).data('hazard');

        // Update all child checkboxes without triggering individual change events
        $(`.hazard-column.${hazardClass}`).prop('checked', isChecked);

        // Update parent states and column visibility
        updateParentCheckboxes();
        updateColumnVisibility();
    });

    // Individual column checkbox handler
    $('.hazard-column').on('change', function() {
        // Update parent checkbox state
        updateParentCheckboxState($(this));
        // Update column visibility
        updateColumnVisibility();
    });

    // Make show/hide all buttons more reliable using delegated events
    $(document).off('click', '#show-all-columns');
    $(document).on('click', '#show-all-columns', function(e) {
        e.preventDefault();
        $('.hazard-column').prop('checked', true);
        updateParentCheckboxes();
        updateColumnVisibility();
    });

    $(document).off('click', '#hide-all-columns');
    $(document).on('click', '#hide-all-columns', function(e) {
        e.preventDefault();
        $('.hazard-column').prop('checked', false);
        updateParentCheckboxes();
        updateColumnVisibility();
    });

    // Helper function to update a specific parent checkbox state
    function updateParentCheckboxState(changedCheckbox) {
        const hazardClasses = changedCheckbox.attr('class').split(' ').filter(cls =>
            cls !== 'form-check-input' && cls !== 'hazard-column');

        hazardClasses.forEach(function(hazardClass) {
            const parentCheckbox = $(`[data-hazard="${hazardClass}"]`);
            const childCheckboxes = $(`.hazard-column.${hazardClass}`);
            const checkedChilds = $(`.hazard-column.${hazardClass}:checked`);

            parentCheckbox.prop('checked', checkedChilds.length > 0);
            parentCheckbox.prop('indeterminate',
                checkedChilds.length > 0 && checkedChilds.length < childCheckboxes.length);
        });
    }

    // Helper function to update all parent checkbox states
    function updateParentCheckboxes() {
        $('.hazard-group').each(function() {
            const hazardClass = $(this).data('hazard');
            const childCheckboxes = $(`.hazard-column.${hazardClass}`);
            const checkedChilds = $(`.hazard-column.${hazardClass}:checked`);

            $(this).prop('checked', checkedChilds.length > 0);
            $(this).prop('indeterminate',
                checkedChilds.length > 0 && checkedChilds.length < childCheckboxes.length);
        });
    }
});